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
import os
import pickle
import sys
import time
import threading
from typing import Dict, List

import numpy as np
from termcolor import colored
from piper_sdk import *

# Record Setting
PERIOD = 0.2  # seconds
DATA_SAVED_PATH = "datasets"
os.makedirs(DATA_SAVED_PATH, exist_ok=True)


def mdeg_to_rad(v_mdeg: float) -> float:
    """Convert 0.001-degree units to radians.

    Args:
      v_mdeg: Angle value in "milli-degree" units (0.001 deg per unit).

    Returns:
      Angle in radians.
    """
    return v_mdeg * (np.pi / 180_000.0)


def current_state(piper: C_PiperInterface_V2) -> np.ndarray:
    """Read one robot state sample.

    The SDK reports:
      - Position in 0.001 mm units (converted to meters).
      - Orientation in 0.001 degree units.
      - Gripper angle in 0.001 degree units (kept per user's original conversion).

    Returns a vector:
      [x(m), y(m), z(m), rx(rad), ry(rad), rz(rad), gripper(rad)]

    Args:
      piper: Connected Piper interface instance.

    Returns:
      A NumPy array of shape (7,) with the state values.
    """
    e = piper.GetArmEndPoseMsgs().end_pose
    g = piper.GetArmGripperMsgs().gripper_state

    # Position: 0.001 mm -> m (1e-6)
    x, y, z = e.X_axis * 1e-6, e.Y_axis * 1e-6, e.Z_axis * 1e-6

    # Orientation: 0.001 deg -> rad via helper
    rx, ry, rz = mdeg_to_rad(e.RX_axis), mdeg_to_rad(e.RY_axis), mdeg_to_rad(e.RZ_axis)

    # Gripper: keep user's original conversion (0.001 deg * 1e-5 -> rad).
    # NOTE: This follows your provided code exactly.
    grip = g.grippers_angle * 1e-5

    arm_state = np.array([x, y, z, rx, ry, rz, grip], dtype=np.float64)
    return arm_state


def sampler_loop(
    piper: C_PiperInterface_V2,
    record_on: threading.Event,
    quit_on: threading.Event,
    lock: threading.Lock,
    ctx: dict,
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
            next_tick = now
            continue

        if now < next_tick:
            time.sleep(next_tick - now)

        sample = current_state(piper)
        with lock:
            ctx["current"].append(sample)

        next_tick += PERIOD


def main() -> None:
    """Entry point: connect, run keyboard loop, and save trajectories to PKL."""
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()

    print(colored("b=Start Record; n=Save Trajectory; m=Reject Trajectory; q=Quit System", "cyan"))

    trajectories: Dict[str, List[np.ndarray]] = {}
    idx = 0

    # Shared state with the sampler thread.
    ctx = {"current": []}
    lock = threading.Lock()
    record_on = threading.Event()
    quit_on = threading.Event()

    th = threading.Thread(
        target=sampler_loop,
        args=(piper, record_on, quit_on, lock, ctx),
        daemon=True,
    )
    th.start()

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "b":
                with lock:
                    ctx["current"] = []  # new buffer for the incoming trajectory
                record_on.set()
                print(colored(f"Recording: tra_{idx}", "green"))

            elif cmd == "n":
                with lock:
                    record_on.clear()
                    to_save = ctx["current"]        # take the filled buffer
                    ctx["current"] = []             # give sampler a fresh buffer
                trajectories[f"tra_{idx}"] = to_save
                print(colored(f"Saved: tra_{idx} ({len(to_save)} states)", "yellow"))
                idx += 1

            elif cmd == "m":
                with lock:
                    record_on.clear()
                    ctx["current"] = []             # drop current buffer
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

    # Summary
    print(colored(f"\nTotal trajectories saved: {len(trajectories)}", "blue"))
    for k, v in trajectories.items():
        print(f"  {k}: {len(v)} states")

    # Ask for a user-provided name and save trajectories as a PKL file.
    if len(trajectories.keys()) != 0:
        name = input("\nEnter a name for this trajectory set (file will be saved as .pkl): ").strip()
        if not name:
            name = "trajectories"
        if not name.lower().endswith(".pkl"):
            name = f"{name}.pkl"

        filepath = os.path.join(DATA_SAVED_PATH, name)
        with open(filepath, "wb") as f:
            pickle.dump(trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(colored(f"Trajectories saved to: {name}", "cyan"))


if __name__ == "__main__":
    main()
