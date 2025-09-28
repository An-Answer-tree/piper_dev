import copy
import numpy as np
import os
import pickle
import sys
import time
import threading
from typing import Dict, List
from termcolor import colored
import cv2

from piper_sdk import *
from pyorbbecsdk import *
from piper_dev.utils import (
    frame_to_bgr_image, 
    connect_camera, 
)
from piper_dev.utils import (
    current_state
)

PERIOD = 0.2  # seconds

def sample_rgb_loop(
    pipeline, 
    record_on: threading.Event,
    quit_on: threading.Event,
    lock: threading.Lock,
    rgb: list
):
    next_tick = None
    while not quit_on.is_set():
        if not record_on.is_set():
            next_tick = None
            time.sleep(0.01)    
            continue        
            
        now = time.perf_counter()
        if next_tick is None:
            next_tick = now + PERIOD
            continue
        
        if now < next_tick:
            time.sleep(next_tick - now)
        
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            print(f"{time.perf_counter():.2f}")
                
            # color_image = frame_to_bgr_image(color_frame)
            
            
            with lock:
                rgb.append(color_frame)

            break
        
        
        
        next_tick += PERIOD
        



def main() -> None:
    # Connect Camera
    orbbec = connect_camera()
    
    rgb = []
    lock = threading.Lock()
    record_on = threading.Event()
    quit_on = threading.Event()
    
    th = threading.Thread(
        target=sample_rgb_loop,
        args=(orbbec, record_on, quit_on, lock, rgb),
        daemon=True,
    )
    th.start()


    record_on.set()
    
    
    
    th.join()
    
if __name__ == "__main__":
    main()