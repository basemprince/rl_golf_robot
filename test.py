"""This script tests the FrankaIkGolfCourseEnv-v0 environment using the SAIClient."""

import time

import cv2
import numpy as np
from sai_rl import SAIClient

from main import FrankaFK, GolfRewardWrapper, is_club_dropped


def get_keyboard_action():  # pylint: disable=too-many-branches
    """Get 7-length action from keyboard input"""
    joint_action = [0.0] * 7
    pressed_key = cv2.waitKey(1) & 0xFF

    speed = 0.3  # Speed for joint movements
    # Joint controls (1-7 keys)
    if pressed_key == ord("d"):
        joint_action[0] = speed
    elif pressed_key == ord("a"):
        joint_action[0] = -speed
    elif pressed_key == ord("w"):
        joint_action[1] = speed
    elif pressed_key == ord("s"):
        joint_action[1] = -speed
    elif pressed_key == ord("r"):
        joint_action[2] = speed
    elif pressed_key == ord("f"):
        joint_action[2] = -speed
    elif pressed_key == ord("t"):
        joint_action[3] = speed
    elif pressed_key == ord("g"):
        joint_action[3] = -speed
    elif pressed_key == ord("y"):
        joint_action[4] = speed
    elif pressed_key == ord("h"):
        joint_action[4] = -speed
    elif pressed_key == ord("u"):
        joint_action[5] = speed
    elif pressed_key == ord("j"):
        joint_action[5] = -speed
    elif pressed_key == ord("i"):
        joint_action[6] = speed
    elif pressed_key == ord("k"):
        joint_action[6] = -speed

    return joint_action, pressed_key


# === Connect and create env ===
sai = SAIClient("FrankaIkGolfCourseEnv-v0")
env = sai.make_env(render_mode="rgb_array")
print("env_metadata", env.metadata)
franka = FrankaFK(env.spec)
env = GolfRewardWrapper(env, include_velocities=True, fk_solver=franka)
obs, info = env.reset()
print("Observation shape:", obs.shape)
print("unwrapped", env.spec)
# Initialize virtual end-effector position
ee_pos = np.array([0.0, 0.0, 0.0])

print("\nControls:")
print("1/q: Joint 1 +/-  2/w: Joint 2 +/-  3/e: Joint 3 +/-")
print("4/r: Joint 4 +/-  5/t: Joint 5 +/-  6/y: Joint 6 +/-  7/u: Joint 7 +/-")
print("ESC: Quit\n")

for step in range(9999999999999999999):
    action, key = get_keyboard_action()
    if key == 27:  # ESC key
        break
    GRIPPER_LOC = action[6]
    obs, reward, terminated, truncated, info = env.step(action)
    gripper_width = obs[7]
    club_pos = obs[21:24]
    club_quat = obs[24:28]
    ee_pos, ee_rot = franka.compute_ee_pose(obs[0:7])  # Forward kinematics for EE
    roll, pitch, yaw = franka.rotation_to_rpy(ee_rot)
    club_dropped = is_club_dropped(club_quat, club_pos, height_threshold=0.11)
    club_lifted = club_pos[2] > 0.14
    is_grasping = 0.005 < gripper_width < 0.015
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
        TEXT3 = f"Club Pos: {club_pos}, Dropped: {club_dropped}"
        text4 = f"Club Lifted: {club_lifted}, Grasping:{is_grasping} Gripper: {obs[7]:0.8f}, {obs[8]:0.8f}"
        cv2.putText(frame, text1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, TEXT3, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"reward: {reward:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, text4, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Franka EE Debug", frame)

    time.sleep(0.1)
    if terminated or truncated:
        obs, info = env.reset()
        ee_pos = np.array([0.0, 0.0, 0.0])  # Reset tracker

env.close()
