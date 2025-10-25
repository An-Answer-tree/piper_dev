import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from pprint import pprint
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds



DATASET_NAME = "my_dataset:0.1.0"
DATA_DIR = "/home/szliutong/tensorflow_datasets"

dataset = tfds.load(name=DATASET_NAME, data_dir=DATA_DIR, split='train', shuffle_files=False)
print(dataset)


for episode in dataset.take(1):
    steps = episode["steps"]
    for step in steps.take(1):
        pprint(step.keys())
        pprint(step["action"].keys())
        pprint(step["observation"].keys())