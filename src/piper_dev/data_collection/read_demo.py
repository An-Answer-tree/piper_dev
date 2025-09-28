import cv2
import json
import numpy as np
import pickle
import PIL
from termcolor import colored

file_path = "dataset/test.pkl"

with open(file_path, 'rb') as f:
    demos = pickle.load(f)
    
# Show demos' structure
print(colored(f"{'*' * 100}", 'yellow'))

print(colored(f"Total structure: ", 'red'))
print(colored(f"demos has keys: {demos.keys()}", 'cyan'))
print(colored(f"For each key, Value has keys: {demos["demo_0"].keys()}", 'cyan'))

print(colored(f"{'*' * 100}", 'yellow'))

print(colored(f"Data insight: ", 'red'))
print(colored(f"instruction: {demos["instruction"]}", 'cyan'))
print(colored(f"state shape: {np.array(demos["demo_0"]["state"]).shape}", 'cyan'))
print(colored(f"rgb shape: {np.array(demos["demo_0"]["rgb"]).shape}", 'cyan'))

print(colored(f"{'*' * 100}", 'yellow'))

print(colored(f"Task demo_0 as example: ", 'red'))
print(colored(f"instruction: {demos["instruction"]}", 'cyan'))
print(colored(f"arm_state: {demos["demo_0"]["state"]}", 'cyan'))
demo_video = demos["demo_0"]["rgb"]
print(colored(f"{'*' * 100}", 'yellow'))

for img in demo_video:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Color Viewer", bgr)
    cv2.waitKey(50)


