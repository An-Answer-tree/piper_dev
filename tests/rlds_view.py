import os, sys
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

DATASET_NAME = "droid"
DATA_DIR = "/media/szliutong/HDD-Storage/Dataset"

droid = tfds.load(name=DATASET_NAME, data_dir=DATA_DIR, split='train', shuffle_files=False)
print(droid)