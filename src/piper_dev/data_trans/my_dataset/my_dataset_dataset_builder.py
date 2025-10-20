"""my_dataset dataset."""

import os
import numpy as np
import pickle
import tensorflow_datasets as tfds
import tensorflow as tf

PATH = "/home/szliutong/Project/piper_dev/dataset"

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = tfds.core.Version('0.1.0')
    RELEASE_NOTES = {
            '0.1.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        features = tfds.features.FeaturesDict({
            "episode_metadata": tfds.features.FeaturesDict({
                "file_path": tfds.features.Text(),
                "recording_folderpath": tfds.features.Text(),
            }),
            "steps": tfds.features.Dataset({
                "action": tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                "action_dict": tfds.features.FeaturesDict({
                    "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=tf.float64),
                    "cartesian_velocity": tfds.features.Tensor(shape=(6,), dtype=tf.float64),
                    "gripper_position": tfds.features.Tensor(shape=(1,), dtype=tf.float64),
                    "gripper_velocity": tfds.features.Tensor(shape=(1,), dtype=tf.float64),
                    "joint_position": tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                    "joint_velocity": tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                }),
                "discount": tfds.features.Scalar(dtype=tf.float32),
                "is_first": tf.bool,
                "is_last": tf.bool,
                "is_terminal": tf.bool,
                "language_instruction": tfds.features.Text(),
                "language_instruction_2": tfds.features.Text(),
                "language_instruction_3": tfds.features.Text(),
                "observation": tfds.features.FeaturesDict({
                    "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=tf.float64),
                    "exterior_image_1_left": tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
                    "exterior_image_2_left": tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
                    "gripper_position": tfds.features.Tensor(shape=(1,), dtype=tf.float64),
                    "joint_position": tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                    "wrist_image_left": tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
                }),
                "reward": tfds.features.Scalar(dtype=tf.float32),
            }),
        })
  
        return self.dataset_info_from_configs(
                features=features, 
                # If there's a common (input, target) tuple from the
                # features, specify them here. They'll be used if
                # `as_supervised=True` in `builder.as_dataset`.
                supervised_keys=None,  # Set to `None` to disable
                homepage='',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        paths = []
        dir = os.path.abspath(PATH)
        for i in os.listdir(dir):
            paths.append(os.path.join(dir, i))

        return {
                'train': self._generate_examples(paths),
        }

    def _generate_examples(self, paths):
        key = 0
        for task_pkl in paths:
            with open(task_pkl, "rb") as f:
                demos = pickle.load(f)
            
            instruction = demos.get("instruction", "")
            
            demo_num = len(demos.keys()) - 1
            for id in range(0, demo_num):
                demo_key = f"demo_{id}"
                demo = demos[demo_key]
                states = self._as_np(demo["state"], dtype=np.float64)   # (T, 7)
                rgbs   = self._as_np(demo["rgb"], dtype=np.uint8)       # (T, H, W, 3)
                
                T = int(min(len(states), len(rgbs)))
                if T == 0:
                    raise RuntimeError(f"In task \"{task_pkl}\", demo_{id} has no data!")
                states = states[:T]
                rgbs = rgbs[:T]

                # Add Content
                episode_metadata = {
                    "file_path": task_pkl,
                    "recording_folderpath": os.path.dirname(task_pkl),
                }
                example = {
                    "episode_metadata": episode_metadata,
                    "steps": self.steps_iter(T, states, rgbs, instruction),
                }
                
                yield key, example
                key += 1

    def _as_np(self, a, dtype=None):
        arr = np.array(a)
        return arr.astype(dtype) if (dtype is not None and arr.dtype != dtype) else arr
    
    def steps_iter(self, T, states, rgbs, instruction):
        zeros1 = np.zeros((1,),  dtype=np.float64)
        zeros6 = np.zeros((6,),  dtype=np.float64)
        zeros7 = np.zeros((7,),  dtype=np.float64)
        
        for t in range(T):
            pose = states[t]
            image = rgbs[t]
            
            # Fill data
            action = pose
            action_dict = {
                "cartesian_position": zeros6,
                "cartesian_velocity": zeros6,
                "gripper_position":   zeros1,
                "gripper_velocity":   zeros1,
                "joint_position":     pose,
                "joint_velocity":     zeros7,
            }
            observation = {
                "cartesian_position": zeros6,
                "exterior_image_1_left": image,
                "exterior_image_2_left": image,
                "gripper_position":   zeros1,
                "joint_position":     zeros7,
                "wrist_image_left":   image,
            }
            
            yield {
                "action": action,
                "action_dict": action_dict,
                "discount": np.float32(1.0),
                "is_first": (t == 0),
                "is_last":  (t == T - 1),
                "is_terminal": (t == T - 1), 
                "language_instruction":   instruction,
                "language_instruction_2": "",
                "language_instruction_3": "",
                "observation": observation,
                "reward": np.float32(0.0),       # if no reward, default 0
            }
            
        