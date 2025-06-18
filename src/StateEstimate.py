import os

# Path to your image folder
folder = "/home/mshashank02/CubeStateEsimator/pose_dataset_HandManipulateBlock_ContinuousTouchSensors-v1_620000"

# Count .png files
num_images = sum(1 for f in os.listdir(folder) if f.endswith(".png"))

print(f"Total number of images: {num_images}")
