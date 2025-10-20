import cv2
import json
import numpy as np
import pickle
import PIL
from termcolor import colored

file_path = "/home/szliutong/Project/piper_dev/dataset/pick the origin block to the green bowl..pkl"

with open(file_path, 'rb') as f:
    demos = pickle.load(f)