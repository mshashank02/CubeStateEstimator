import os
import json
import numpy as np
import mujoco
import gymnasium as gym
import gymnasium_robotics
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.utils import get_schedule_fn
from PIL import Image
from tqdm import trange
import mujoco.viewer

# Configuration
ENV_ID = "HandManipulateBlock_ContinuousTouchSensors-v1"
N_EPISODES, MAX_STEPS = 500, 200
W, H = 256, 256
SAVE_DIR = f"dataset_{ENV_ID}"
os.makedirs(SAVE_DIR + "/rgb", exist_ok=True)
os.makedirs(SAVE_DIR + "/depth", exist_ok=True)

# Setup environment
def make_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = Monitor(env)
    env = TimeFeatureWrapper(env)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load("/home/mshashank02/CubeStateEsimator/vecnorm_600000.pkl", venv)
    venv.training, venv.norm_reward = False, False
    return venv

env = make_env()
model = TQC.load("/home/mshashank02/CubeStateEsimator/best_model_620000_steps.zip", env=env, device="cpu",
                 custom_objects={"lr_schedule": get_schedule_fn(1e-3)})

raw = env.envs[0]
while hasattr(raw, "env"):
    raw = raw.env
model_mj, data = raw.model, raw.data

# Set up offscreen OpenGL context
ctx = mujoco.GLContext(W, H)
ctx.make_current()

scene = mujoco.MjvScene(model_mj, maxgeom=1000)
ctx_mjr = mujoco.MjrContext(model_mj, mujoco.mjtFontScale.mjFONTSCALE_150)

# Use the fixed camera settings from XML
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE

# Camera position
cam_pos = np.array([0.451, 0.437, 0.507])
xyaxes = np.array([
    [0.698, -0.716, -0.000],  # x-axis
    [0.207,  0.202,  0.957],  # y-axis
])
zaxis = np.cross(xyaxes[0], xyaxes[1])  # z-axis = x × y

# Make it look "forward" along zaxis
lookat = cam_pos + zaxis * 0.3

cam.lookat[:] = lookat
cam.distance = np.linalg.norm(cam_pos - lookat)
cam.azimuth = np.rad2deg(np.arctan2(zaxis[1], zaxis[0]))
cam.elevation = np.rad2deg(np.arcsin(zaxis[2] / np.linalg.norm(zaxis)))


# Launch passive viewer
viewer = mujoco.viewer.launch_passive(model_mj, data)
viewer.cam.type = cam.type
viewer.cam.lookat[:] = cam.lookat
viewer.cam.distance = cam.distance
viewer.cam.azimuth = cam.azimuth
viewer.cam.elevation = cam.elevation

near, far = model_mj.vis.map.znear, model_mj.vis.map.zfar

dataset = []
for ep in trange(N_EPISODES):
    obs = env.reset()
    tid = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, "target")
    for gi in range(model_mj.ngeom):
        if model_mj.geom_bodyid[gi] == tid:
            model_mj.geom_rgba[gi] = [1, 1, 1, 0]

    for t in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        mujoco.mjv_updateScene(model_mj, data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, W, H), scene, ctx_mjr)

        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        depth_raw = np.zeros((H, W), dtype=np.float32)
        mujoco.mjr_readPixels(rgb, depth_raw, mujoco.MjrRect(0, 0, W, H), ctx_mjr)

        depth = near * far / (far - (far - near) * depth_raw)

        pos = data.body("object").xpos.copy()
        quat = data.body("object").xquat.copy()

        rgb_path = f"{SAVE_DIR}/rgb/ep{ep:03d}_step{t:03d}.png"
        depth_path = f"{SAVE_DIR}/depth/ep{ep:03d}_step{t:03d}.npy"

        Image.fromarray(rgb[::-1]).save(rgb_path)
        np.save(depth_path, depth)

        dataset.append({"rgb": rgb_path, "depth": depth_path,
                        "position": pos.tolist(), "quaternion": quat.tolist()})

        viewer.sync()

        print(f"Saved {rgb_path}", pos, quat)
        if done[0]:
            break

viewer.close()

with open(SAVE_DIR + "/poses.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("\u2714\ufe0f Data collection complete")
