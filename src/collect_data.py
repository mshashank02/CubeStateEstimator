import os
import gymnasium as gym
import gymnasium_robotics
import numpy as np
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.utils import get_schedule_fn
from tqdm import trange
from PIL import Image
import json
import mujoco

# --- Config ---
ENV_ID = "HandManipulateBlock_ContinuousTouchSensors-v1"
SEED = 1
STEPS = 620000
N_EPISODES = 500
MAX_STEPS = 200
SAVE_DIR = f"pose_dataset_1_{ENV_ID}_{STEPS}"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Setup environment with correct wrappers ---
def make_env():
    base_env = gym.make(ENV_ID, render_mode="rgb_array")
    base_env = Monitor(base_env)
    base_env = TimeFeatureWrapper(base_env)
    env = DummyVecEnv([lambda: base_env])
    env = VecNormalize.load("/home/mshashank02/CubeStateEsimator/vecnorm_600000.pkl", env)
    env.training = False
    env.norm_reward = False
    return env

env = make_env()

# --- Load model with safe learning rate fallback ---
model = TQC.load(
    "/home/mshashank02/CubeStateEsimator/best_model_620000_steps.zip",
    env=env,
    device="cpu",
    custom_objects={"lr_schedule": get_schedule_fn(1e-3)}
)

# --- Dataset buffer ---
poses = []

# --- Data collection ---
for ep in trange(N_EPISODES, desc="Collecting episodes"):
    obs = env.reset()

    # 🔥 Hide the translucent goal cube after reset
    raw_env = env.envs[0]
    while hasattr(raw_env, "env"):
        raw_env = raw_env.env

    target_body_id = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_BODY, "target")
    for geom_id in range(raw_env.model.ngeom):
        if raw_env.model.geom_bodyid[geom_id] == target_body_id:
            raw_env.model.geom_rgba[geom_id, :] = np.array([1, 1, 1, 0])  # ✅ fully transparent


    for t in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Render image for dataset
        rgb_array = env.render(mode="rgb_array")
        env.render(mode="human")  # optional live render

        # Unwrap to get pose
        raw_env = env.envs[0]
        while hasattr(raw_env, "env"):
            raw_env = raw_env.env

        model = raw_env.model
        default_cam_id = -1  # Gym uses camera index 0 by default for rendering

        print("📷 Active Camera Configuration:")
        print(f"  cam_pos     : {model.cam_pos[default_cam_id]}")
        print(f"  cam_quat    : {model.cam_quat[default_cam_id]}")
        print(f"  cam_fovy    : {model.cam_fovy[default_cam_id]}°")
        print(f"  cam_type    : {model.cam_type[default_cam_id]}")
        print(f"  cam_trackid : {model.cam_trackid[default_cam_id]}")

        pos = raw_env.data.body("object").xpos.copy()
        quat = raw_env.data.body("object").xquat.copy()

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

# --- Save metadata ---
json_path = f"{SAVE_DIR}/poses.json"
with open(json_path, "w") as f:
    json.dump(poses, f, indent=2)

print(f"\n✅ Dataset collection complete!")
print(f"Total datapoints (image-pose pairs): {len(poses)}")
print(f"Dataset saved to folder: {SAVE_DIR}")
print(f"Pose metadata saved to: {json_path}")
