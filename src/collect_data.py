import os
import gymnasium as gym
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from gymnasium.wrappers import RecordEpisodeStatistics
from tqdm import trange
from PIL import Image
import json

# --- Config ---
ENV_ID = "HandManipulateBlockRotateXYZ-v1"
SEED = 1
STEPS = 620000
N_EPISODES = 500  # Increased to ~100,000 images
MAX_STEPS = 200
SAVE_DIR = f"pose_dataset_{ENV_ID}_{STEPS}"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Setup environment with RGB rendering ---
def make_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env.reset(seed=SEED)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(f"models/{ENV_ID}/vecnorm_{STEPS}.pkl", env)
    env.training = False
    env.norm_reward = False
    return env

env = make_env()

# Load trained model
model = TQC.load(f"models/{ENV_ID}/best_model_{STEPS}_steps.zip", env=env, device="cuda")

# --- Dataset buffers ---
poses = []
images = []

for t in range(MAX_STEPS):
    # Predict action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # Get RGB frame for saving
    rgb_array = env.render(mode="rgb_array")

    # Render live simulation
    env.render(mode="human")

    # Extract cube pose from MuJoCo sim
    sim = env.envs[0].unwrapped.sim
    pos = sim.data.get_body_xpos("object").copy()
    quat = sim.data.get_body_xquat("object").copy()

    # Save image
    img_filename = f"{SAVE_DIR}/ep{ep:03d}_step{t:03d}.png"
    Image.fromarray(rgb_array).save(img_filename)

    # Save pose
    poses.append({
        "image": img_filename,
        "position": pos.tolist(),
        "quaternion": quat.tolist()
    })

    print(f"Saved datapoint: {img_filename} | pos: {np.round(pos, 3)} | quat: {np.round(quat, 3)}")

    if done[0]:
        break


# --- Save pose metadata ---
json_path = f"{SAVE_DIR}/poses.json"
with open(json_path, "w") as f:
    json.dump(poses, f, indent=2)

print(f"\n✅ Dataset collection complete!")
print(f"Total datapoints (image-pose pairs): {len(poses)}")
print(f"Dataset saved to folder: {SAVE_DIR}")
print(f"Pose metadata saved to: {json_path}")
