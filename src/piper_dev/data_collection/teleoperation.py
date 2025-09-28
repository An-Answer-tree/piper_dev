# -*- coding: utf-8 -*-
"""Keyboard-driven trajectory recorder.

This script records robot arm states and synchronized camera frames using two
threads coordinated by events:

Hotkeys:
  - b: Start recording into a new ``demo_*``
  - n: Save current trajectory
  - m: Reject (drop) current trajectory
  - q: Quit

Attributes:
  PERIOD (float): Sampling period in seconds.
  DATA_SAVED_PATH (str): Directory where the dataset pickle is saved.
"""

import copy
import os
import pickle
import threading
import time
from termcolor import colored
from typing import List

import numpy as np
from piper_sdk import *
from pyorbbecsdk import *
from piper_dev.utils import connect_camera, current_state
from piper_dev.utils import frame_to_bgr_image, bgrs_to_rgbs

from piper_dev.data_collection.config import TeleCFG

# ===== Settings =====
config = TeleCFG()
PERIOD = config.period          # Sampling period (seconds)
DATA_SAVED_PATH = config.saved_path
os.makedirs(DATA_SAVED_PATH, exist_ok=True)
# Camera setting
WIDTH = config.width
HEIGHT = config.height
FPS = config.fps


def state_loop_tick_broadcast(
    piper: C_PiperInterface_V2,
    record_on: threading.Event,
    quit_on: threading.Event,
    # events
    tick_evt: threading.Event,     # Arm -> (Arm & Cam): broadcast "tick" to start this shot
    cam_done: threading.Event,     # Cam -> Arm: ack when the camera frame for this shot is ready
    # buffer
    buf_lock: threading.Lock,
    state: list,                   # Append: robot state samples for each shot
) -> None:
    """Arm thread loop: fixed-rate driver with tick broadcast and camera ack.

    The arm thread maintains a drift-free fixed-rate schedule. At each shot, it
    broadcasts a ``tick`` event to trigger both the arm sampling and the camera
    capture nearly simultaneously, appends the arm sample to the buffer, and
    then waits for the camera acknowledgment before advancing to the next tick.

    Args:
      piper: Connected robot arm interface.
      record_on: Event toggling recording on/off.
      quit_on: Event signaling both threads to exit.
      tick_evt: Event used to broadcast the start of a shot.
      cam_done: Event set by the camera when its frame for the shot is ready.
      buf_lock: Mutex protecting access to ``state``.
      state: List buffer to append arm samples to (one per shot).

    Returns:
      None
    """
    next_tick = None
    while not quit_on.is_set():
        if not record_on.is_set():
            next_tick = None
            time.sleep(0.01)
            continue

        # Drift-free schedule: sleep until the next tick boundary.
        now = time.perf_counter()
        if next_tick is None:
            # First shot starts on the next period boundary (adjust as needed).
            next_tick = now + PERIOD
        if now < next_tick:
            time.sleep(next_tick - now)

        # 1) Broadcast "tick": arm & camera start this shot near-simultaneously.
        tick_evt.set()

        # 2) Sample the arm.
        sample = current_state(piper)
        print(f"time_1: {time.perf_counter():.2f}")

        # 3) Append the arm sample.
        with buf_lock:
            state.append(sample)

        # 4) Wait for the camera to finish this shot (prevents the arm advancing early).
        cam_done.wait()
        cam_done.clear()

        # 5) Schedule the next shot.
        next_tick += PERIOD


def rgb_loop_tick_broadcast(
    pipeline,
    record_on: threading.Event,
    quit_on: threading.Event,
    # events
    tick_evt: threading.Event,     # Arm -> (Arm & Cam): broadcast "tick"
    cam_done: threading.Event,     # Cam -> Arm: ack after a frame is captured
    # buffer
    buf_lock: threading.Lock,
    rgb: list,                     # Append: BGR frames (numpy arrays) per shot
) -> None:
    """Camera thread loop: wait for tick, capture one frame, then ack.

    The camera thread blocks on the broadcast ``tick`` event, consumes it,
    captures exactly one color frame (retrying by ``wait_for_frames`` until a
    valid frame is returned), appends the frame to the buffer, and then sets
    ``cam_done`` to let the arm proceed to the next shot.

    Args:
      pipeline: Initialized camera/pipeline handle.
      record_on: Event toggling recording on/off.
      quit_on: Event signaling both threads to exit.
      tick_evt: Event set by the arm to start a shot.
      cam_done: Event set by the camera when the frame is ready.
      buf_lock: Mutex protecting access to ``rgb``.
      rgb: List buffer to append converted BGR frames to (one per shot).

    Returns:
      None
    """
    while not quit_on.is_set():
        if not record_on.is_set():
            time.sleep(0.01)
            continue

        # Wait for the "tick" and consume it to process this shot exactly once.
        tick_evt.wait()
        tick_evt.clear()

        # Capture a single frame; strictly wait until a valid color frame is obtained.
        color_frame = None
        while not quit_on.is_set() and record_on.is_set():
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            cf = frames.get_color_frame()
            if cf is not None:
                print(f"time_2: {time.perf_counter():.2f}")
                color_frame = frame_to_bgr_image(cf)
                break

        if color_frame is None:
            # Device stopped or transient failure: do not ack; arm will keep waiting or the user will stop.
            continue

        # Append the camera frame.
        with buf_lock:
            rgb.append(color_frame)

        # Ack completion so the arm can advance to the next shot.
        cam_done.set()


def main() -> None:
    """Program entry point.

    Connects to the robot arm and camera, starts the worker threads, runs the
    keyboard-driven loop for recording/ending/rejecting trajectories, and saves
    the dataset as a pickle at the end.

    Hotkeys:
      - b: Start a new recording session into ``demo_{idx}``.
      - n: Save the current session's buffers into ``demos``.
      - m: Reject (clear) the current buffers without saving.
      - q: Quit the program.

    Returns:
      None
    """
    # Connect Robotic Arm
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    # Connect Camera
    orbbec = connect_camera(width=WIDTH, height=HEIGHT, fps = FPS)

    print(colored("Change Config in 'config.py'", "yellow"))
    print(colored("b=Start Record; n=Save Trajectory; m=Reject Trajectory; q=Quit System", "cyan"))

    demos = {}
    idx = 0

    state: List = []
    rgb: List = []
    buf_lock = threading.Lock()

    record_on = threading.Event()
    quit_on = threading.Event()

    # Events: shared tick broadcast + camera completion ack.
    tick_evt = threading.Event()
    cam_done = threading.Event()

    th_state = threading.Thread(
        target=state_loop_tick_broadcast,
        args=(piper, record_on, quit_on, tick_evt, cam_done, buf_lock, state),
        daemon=True,
    )
    th_rgb = threading.Thread(
        target=rgb_loop_tick_broadcast,
        args=(orbbec, record_on, quit_on, tick_evt, cam_done, buf_lock, rgb),
        daemon=True,
    )
    th_state.start()
    th_rgb.start()

    try:
        while True:
            cmd = input("> ").strip().lower()
            if cmd == "b":
                demos[f"demo_{idx}"] = {}
                with buf_lock:
                    state.clear()
                    rgb.clear()

                # Reset per-session events and start recording.
                tick_evt.clear()
                cam_done.clear()
                record_on.set()
                print(colored(f"Recording: demo_{idx}", "green"))

            elif cmd == "n":
                # Stop recording and copy buffers for saving.
                record_on.clear()
                tick_evt.set()
                cam_done.set()

                with buf_lock:
                    to_save_state = copy.deepcopy(state)
                    # Convert BGR frames to RGB before saving (safer for most consumers).
                    to_save_rgb = bgrs_to_rgbs(copy.deepcopy(rgb))
                    state.clear()
                    rgb.clear()

                # Safety trim (should already match, but keep the invariant).
                Ls, Lr = len(to_save_state), len(to_save_rgb)
                if Ls != Lr:
                    L = min(Ls, Lr)
                    print(colored(f"Trim tails to align: state={Ls} rgb={Lr} -> {L}", "magenta"))
                    to_save_state = to_save_state[:L]
                    to_save_rgb = to_save_rgb[:L]

                demos[f"demo_{idx}"]["state"] = to_save_state
                demos[f"demo_{idx}"]["rgb"] = to_save_rgb
                print(colored(f"Saved: demo_{idx} ({len(to_save_state)} states, {len(to_save_rgb)} frames)", "yellow"))
                idx += 1

            elif cmd == "m":
                # Stop recording and discard current buffers.
                record_on.clear()
                tick_evt.set()
                cam_done.set()
                with buf_lock:
                    state.clear()
                    rgb.clear()
                print(colored("Rejected current trajectory.", "magenta"))

            elif cmd == "q":
                # Quit: stop recording and wake any waiting threads.
                record_on.clear()
                quit_on.set()
                tick_evt.set()
                cam_done.set()
                break
            else:
                print("Use: b / n / m / q")

    except KeyboardInterrupt:
        quit_on.set()
        tick_evt.set()
        cam_done.set()

    th_rgb.join()
    th_state.join()

    piper.DisconnectPort()
    orbbec.stop()

    print(colored(f"\nTotal demos saved: {len(demos)}", "blue"))

    if demos:
        name = input("\nEnter a language instruction for this demo set (file will be saved as .pkl): ").strip() or "demo"
        demos["instruction"] = name
        name = f"{name}.pkl"
        filepath = os.path.join(DATA_SAVED_PATH, name)
        with open(filepath, "wb") as f:
            pickle.dump(demos, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(colored(f"Demos saved to: {name}", "cyan"))


if __name__ == "__main__":
    main()
