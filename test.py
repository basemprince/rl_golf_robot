"""This script tests the FrankaIkGolfCourseEnv-v0 environment using the SAIClient."""

import time

import cv2
import numpy as np
from sai_rl import SAIClient

from main import FrankaFK  # Import the FrankaFK class from main.py

# === Connect and create env ===
sai = SAIClient("FrankaIkGolfCourseEnv-v0")
env = sai.make_env(render_mode="rgb_array")
print("env_metadata", env.metadata)
obs, info = env.reset()
franka = FrankaFK(env.spec)
print("Observation shape:", obs.shape)
print("unwrapped", env.spec)
# Initialize virtual end-effector position
ee_pos = np.array([0.0, 0.0, 0.0])

for step in range(1250):
    action = env.action_space.sample()  # Random action
    action = [0, 0, 0, -0.01, 0, 0, 0]  # Zero action for testing
    obs, reward, terminated, truncated, info = env.step(action)

    ee_pos, ee_rot = franka.compute_ee_pose(obs[0:7])  # Forward kinematics for EE
    roll, pitch, yaw = franka.rotation_to_rpy(ee_rot)
    # Update virtual EE position by adding displacement from action
    ee_pos += action[:3]

    # Compute average gripper position (absolute)
    avg_gripper = (obs[7] + obs[8]) / 2

    print(f"\nStep {step}")
    print("Action Δxyz:", action[:3])
    print("Virtual EE Pos:", ee_pos)
    print("Avg Gripper Pos:", avg_gripper)
    print("Observation:", obs)

    frame = env.render()

    if frame is not None:
        # Ensure numpy array and convert to BGR for OpenCV
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        # Convert float (0–1) to uint8 (0–255)
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Remove alpha channel if exists
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]

        # Convert RGB → BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Now overlay text
        text1 = f"EE Pos: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}"
        text2 = f"EE PRY: P={roll:.2f}, R={pitch:.2f}, Y={yaw:.2f}"
        cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Franka EE Debug", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    time.sleep(0.5)
    if terminated or truncated:
        obs, info = env.reset()
        ee_pos = np.array([0.0, 0.0, 0.0])  # Reset tracker

env.close()
