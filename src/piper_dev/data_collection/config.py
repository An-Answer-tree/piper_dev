from dataclasses import dataclass

@dataclass
class TeleCFG:
    # Record data per "period"; Record frequency = 1 / period
    period = 0.1              # 20Hz
    saved_path = "./dataset"   # Path to save data
    
    # Camera record setting
    width = 640
    height = 480
    fps = 30