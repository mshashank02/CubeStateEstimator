import os
import json

with open("pose_dataset_HandManipulateBlock_ContinuousTouchSensors-v1_620000/poses.json") as f:
    data = json.load(f)

missing = [item["image"] for item in data if not os.path.isfile(item["image"])]

print(f"Missing images: {len(missing)}")
for img in missing[:10]:
    print(img)
