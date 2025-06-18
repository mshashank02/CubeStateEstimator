import os
import json

BASE_DIR = "/home/shashank/CubeStateEstimator"  # or wherever the dataset is actually located

with open(os.path.join(BASE_DIR, "pose_dataset_HandManipulateBlock_ContinuousTouchSensors-v1_620000/poses.json")) as f:
    data = json.load(f)

missing = []
for item in data:
    img_path = os.path.join(BASE_DIR, item["image"])
    if not os.path.isfile(img_path):
        missing.append(item["image"])

print(f"Missing images: {len(missing)}")
for img in missing[:10]:
    print(img)

