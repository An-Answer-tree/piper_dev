# -*- coding: utf-8 -*-
"""Keyboard-driven trajectory recorder with precise 0.2 s sampling.

Hotkeys:
  - b: Start recording into a new 'tra_*'
  - n: Save the current trajectory
  - m: Reject (drop) the current trajectory
  - q: Quit the system

Sampling runs in a background thread at a fixed 0.2 s period using a drift-free
schedule (next_tick += PERIOD). All trajectories are kept in-memory and saved
at the end as a PKL file (pickle) with the user-provided file name.
"""
import copy
import numpy as np
import os
import pickle
import sys
import time
import threading
from typing import Dict, List
from termcolor import colored

from piper_sdk import *
from pyorbbecsdk import *
from piper_dev.utils import (
    frame_to_bgr_image, 
    connect_camera, 
)
from piper_dev.utils import (
    current_state
)

# Record Setting
PERIOD = 0.2  # seconds
DATA_SAVED_PATH = "datasets"
os.makedirs(DATA_SAVED_PATH, exist_ok=True)


def sample_state_loop(
    piper: C_PiperInterface_V2,
    record_on: threading.Event,
    quit_on: threading.Event,
    lock: threading.Lock,
    state: list,
) -> None:
    """Background sampler thread loop (drift-free at PERIOD).

    The loop:
      * Waits for 'record_on' to be set.
      * Samples exactly at 'next_tick' using a monotonic clock.
      * Appends one state to ctx['current'] at each tick.

    Args:
      piper: Connected Piper interface instance.
      record_on: Event signaling recording state (set = recording).
      quit_on: Event signaling the sampler to exit.
      lock: Mutex protecting shared context structures.
      ctx: Shared dict containing 'current': List[np.ndarray] buffer.
    """
    next_tick = None
    while not quit_on.is_set():
        if not record_on.is_set():
            next_tick = None
            time.sleep(0.01)  # idle to reduce CPU
            continue

        now = time.perf_counter()
        if next_tick is None:
            next_tick = now + PERIOD
            continue

        if now < next_tick:
            time.sleep(next_tick - now)
        if now > next_tick:
            print(colored("wrong", "red"))

        sample = current_state(piper)
        with lock:
            state.append(sample)

        next_tick += PERIOD
        

def sample_rgb_loop(
    record_on: threading.Event,
    quit_on: threading.Event,
    lock: threading.Lock,
):
    pass
    


def main() -> None:
    """Entry point: connect, run keyboard loop, and save trajectories to PKL."""

    # Connect Robot Arm
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    
    # Connect Camera
    orbbec = connect_camera()

    print(colored("b=Start Record; n=Save Trajectory; m=Reject Trajectory; q=Quit System", "cyan"))

    # All Demos that are Recorded
    demos = {}
    idx = 0

    # Shared state with the sampler thread.
    state = []
    lock = threading.Lock()
    record_on = threading.Event()
    quit_on = threading.Event()

    th = threading.Thread(
        target=sample_state_loop,
        args=(piper, record_on, quit_on, lock, state),
        daemon=True,
    )
    th.start()
    
    

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "b":
                demos[f"demo_{idx}"] = {}
                with lock:
                    state.clear()  # new buffer for the incoming trajectory
                record_on.set()
                print(colored(f"Recording: demo_{idx}", "green"))

            elif cmd == "n":
                with lock:
                    record_on.clear()
                    to_save = copy.deepcopy(state)        # take the filled buffer
                    state.clear()            # give sampler a fresh buffer
                demos[f"demo_{idx}"]["state"] = to_save
                print(colored(f"Saved: demo_{idx} ({len(to_save)} states)", "yellow"))
                idx += 1

            elif cmd == "m":
                with lock:
                    record_on.clear()
                    state.clear()             # drop current buffer
                print(colored("Rejected current trajectory.", "magenta"))

            elif cmd == "q":
                record_on.clear()
                quit_on.set()
                break

            else:
                print("Use: b / n / m / q")

    except KeyboardInterrupt:
        quit_on.set()

    th.join()
    piper.DisconnectPort()
    orbbec.stop()

    # Summary
    print(colored(f"\nTotal demos saved: {len(demos)}", "blue"))

    # Ask for a user-provided name and save demos as a PKL file.
    if len(demos.keys()) != 0:
        name = input("\nEnter a language instruction for this demo set (file will be saved as .pkl): ").strip()
        if not name:
            name = "demo"
        demos["instruction"] = name
        name = f"{name}.pkl"

        filepath = os.path.join(DATA_SAVED_PATH, name)
        with open(filepath, "wb") as f:
            pickle.dump(demos, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(colored(f"Demos saved to: {name}", "cyan"))
        print(demos)


if __name__ == "__main__":
    main()
