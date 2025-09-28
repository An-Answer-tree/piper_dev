# -*- coding: utf-8 -*-
"""Keyboard-driven trajectory recorder (Scheme A: tick broadcast + camera ack)

- Fixed-rate sampling with drift-free schedule (next_tick += PERIOD)
- tick_evt: Arm broadcasts "tick" so Arm & Camera start this shot near-simultaneously
- cam_done: Camera acks after frame captured; Arm waits this ack to avoid advancing
- No Barrier / tick_id / last_seen / timestamps

Hotkeys:
  - b: Start recording into a new 'demo_*'
  - n: Save current trajectory
  - m: Reject (drop) current trajectory
  - q: Quit
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

# ===== Settings =====
PERIOD = 0.2          # seconds
CAM_WAIT_MS = 100     # wait_for_frames 单次等待(ms)，循环重试直到取到帧
DATA_SAVED_PATH = "datasets"
os.makedirs(DATA_SAVED_PATH, exist_ok=True)


def state_loop_tick_broadcast(
    piper: C_PiperInterface_V2,
    record_on: threading.Event,
    quit_on: threading.Event,
    # events
    tick_evt: threading.Event,     # Arm -> (Arm & Cam): 同步“开拍”触发
    cam_done: threading.Event,     # Cam -> Arm: 本拍帧完成确认
    # buffer
    buf_lock: threading.Lock,
    state: list,                   # append(sample)
) -> None:
    """Arm thread: fixed-rate driver; broadcasts tick; waits for camera ack each shot."""
    next_tick = None
    while not quit_on.is_set():
        if not record_on.is_set():
            time.sleep(0.01)
            continue

        # 漂移校正：到达拍点
        now = time.perf_counter()
        if next_tick is None:
            # 第一拍立即触发（若想对齐到下个周期：next_tick = now + PERIOD）
            next_tick = now + PERIOD
        if now < next_tick:
            time.sleep(next_tick - now)

        # 1) 同步触发“开拍”：Arm & Cam 几乎同时开始各自采集
        tick_evt.set()

        # 2) Arm 采样
        sample = current_state(piper)

        # 3) 缓存 Arm 样本
        with buf_lock:
            state.append(sample)

        # 4) 等待相机完成本拍（严格防止 Arm 超前）
        cam_done.wait()
        cam_done.clear()

        # 5) 下一拍时刻
        next_tick += PERIOD


def rgb_loop_tick_broadcast(
    pipeline,
    record_on: threading.Event,
    quit_on: threading.Event,
    # events
    tick_evt: threading.Event,     # Arm -> (Arm & Cam)
    cam_done: threading.Event,     # Cam -> Arm
    # buffer
    buf_lock: threading.Lock,
    rgb: list,                     # append(color_frame)
) -> None:
    """Camera thread: waits for tick, captures one color frame, then acks cam_done."""
    while not quit_on.is_set():
        if not record_on.is_set():
            time.sleep(0.01)
            continue

        # 等待“开拍”触发；清掉以消费这一次 tick
        tick_evt.wait()
        tick_evt.clear()

        # 取一帧（严格：直到拿到有效 color_frame 才确认）
        color_frame = None
        while not quit_on.is_set() and record_on.is_set():
            frames = pipeline.wait_for_frames(CAM_WAIT_MS)
            if frames is None:
                continue
            cf = frames.get_color_frame()
            if cf is not None:
                color_frame = frame_to_bgr_image(cf)
                break

        if color_frame is None:
            # 设备异常/被停止：不确认，让 Arm 继续等待或主线程中止
            continue

        # 缓存 Camera 帧
        with buf_lock:
            rgb.append(color_frame)

        # 确认完成，允许 Arm 进入下一拍
        cam_done.set()


def main() -> None:
    # 连接设备
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    orbbec = connect_camera()

    print(colored("b=Start Record; n=Save Trajectory; m=Reject Trajectory; q=Quit System", "cyan"))

    demos = {}
    idx = 0

    state: List = []
    rgb: List = []
    buf_lock = threading.Lock()

    record_on = threading.Event()
    quit_on = threading.Event()

    # 事件：公共 tick 触发 + 相机完成确认
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

                # 清事件，开始录制
                tick_evt.clear()
                cam_done.clear()
                record_on.set()
                print(colored(f"Recording: demo_{idx}", "green"))

            elif cmd == "n":
                # 停止录制
                record_on.clear()

                # 拷贝缓冲
                with buf_lock:
                    to_save_state = copy.deepcopy(state)
                    # to_save_rgb = copy.deepcopy(rgb)
                    to_save_rgb = bgrs_to_rgbs(copy.deepcopy(rgb))
                    state.clear()
                    rgb.clear()

                # 安全裁剪（理论上长度应一致）
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
                # 停止并丢弃
                record_on.clear()
                with buf_lock:
                    state.clear()
                    rgb.clear()
                print(colored("Rejected current trajectory.", "magenta"))

            elif cmd == "q":
                # 退出：关录制并唤醒等待线程
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
