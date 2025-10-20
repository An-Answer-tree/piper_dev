import cv2
import numpy as np
import pickle
from termcolor import colored

file_path = "/home/szliutong/Project/piper_dev/dataset/pick the origin block to the green bowl..pkl"

with open(file_path, 'rb') as f:
    demos = pickle.load(f)
    
# Show demos' structure
print(colored(f"{'*' * 100}", 'yellow'))

print(colored(f"Total structure: ", 'red'))
print(colored(f"demos has keys: {demos.keys()}", 'cyan'))
episode_keys = demos["demo_0"].keys()
print(colored(f"For each key, Value has keys: {episode_keys}", 'cyan'))

print(colored(f"{'*' * 100}", 'yellow'))

print(colored(f"Data insight: ", 'red'))
instruction = demos["instruction"]
print(colored(f"instruction: {instruction}", 'cyan'))
demo_0_state = demos["demo_0"]["state"]
demo_0_rgb = demos["demo_0"]["rgb"]
print(colored(f"state shape: {np.array(demo_0_state).shape}", 'cyan'))
print(colored(f"rgb shape: {np.array(demo_0_rgb).shape}", 'cyan'))

print(colored(f"{'*' * 100}", 'yellow'))

print(colored(f"Task demo_0 as example: ", 'red'))
print(colored(f"instruction: {instruction}", 'cyan'))
print(colored(f"arm_state: {demo_0_state}", 'cyan'))
demo_video = demos["demo_0"]["rgb"]
print(colored(f"{'*' * 100}", 'yellow'))

for img in demo_video:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Color Viewer", bgr)
    cv2.waitKey(50)


