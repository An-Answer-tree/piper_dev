import logging
from tqdm import tqdm
import os
import subprocess

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),                      # 输出到终端
        logging.FileHandler("tests/app.log", encoding='utf-8') # 输出到文件
    ]
)

DATASET_DIR = "/media/szliutong/HDD-Storage/Dataset/Open_X_Embodiment/bridge/0.1.0"
CLOUD_DIR = "gs://gresearch/robotics/bridge/0.1.0"

broken_files = []
for i in tqdm(range(922, 1024), desc="Cheack Files"):
    name = f"bridge-train.tfrecord-{i:05d}-of-01024"
    local_file = os.path.join(DATASET_DIR, name)
    cloud_file = os.path.join(CLOUD_DIR, name)
    
    local_crc = subprocess.check_output(
        ["gsutil", "hash", "-c", local_file], text=True
    )
    local_crc = [l for l in local_crc.splitlines() if "crc32c" in l][0].split(":")[1].strip()
    print(local_crc)

    # 取云端 CRC
    cloud_crc = subprocess.check_output(
        ["gsutil", "stat", cloud_file], text=True
    )
    cloud_crc = [l for l in cloud_crc.splitlines() if "crc32c" in l][0].split(":")[1].strip()
    print(cloud_crc)

    if local_crc != cloud_crc:
        logging.info(f"Broken: {name}")
        broken_files.append(name)
        
logging.info(f"PROCESS END: Broken files: {broken_files}")
print(broken_files)
    