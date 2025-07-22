"""This script tests the FrankaIkGolfCourseEnv-v0 environment using the SAIClient."""

import time

import numpy as np
from sai_rl import SAIClient

# === Connect and create env ===
sai = SAIClient("FrankaIkGolfCourseEnv-v0")
env = sai.make_env(render_mode="human")

obs, info = env.reset()
print("Observation shape:", obs.shape)

# Initialize virtual end-effector position
ee_pos = np.array([0.0, 0.0, 0.0])

for step in range(20):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)

    # Update virtual EE position by adding displacement from action
    ee_pos += action[:3]

    # Compute average gripper position (absolute)
    avg_gripper = (obs[7] + obs[8]) / 2

    print(f"\nStep {step}")
    print("Action Δxyz:", action[:3])
    print("Virtual EE Pos:", ee_pos)
    print("Avg Gripper Pos:", avg_gripper)
    print("Observation:", obs)

    time.sleep(0.5)
    if terminated or truncated:
        obs, info = env.reset()
        ee_pos = np.array([0.0, 0.0, 0.0])  # Reset tracker

env.close()
