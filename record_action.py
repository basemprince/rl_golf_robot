"""Module for recording human demonstrations for behavioral cloning."""

from datetime import datetime

import cv2
import numpy as np
from sai_rl import SAIClient
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from main import FrankaFK, GolfRewardWrapper, simplify_obs


def get_keyboard_action():
    """Get 7-length action from keyboard input"""
    joint_action = [0.0] * 7
    pressed_key = cv2.waitKey(1) & 0xFF
    speed = 0.3

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


# === Connect to Env ===
sai = SAIClient("FrankaIkGolfCourseEnv-v0")
env = sai.make_env(render_mode="rgb_array")
franka = FrankaFK(env.spec)
env = GolfRewardWrapper(env, include_velocities=True, fk_solver=franka)
venv = DummyVecEnv([lambda: env])  # Vectorize
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=5.0, clip_reward=10.0)
obs = venv.reset()
ee_pos = np.array([0.0, 0.0, 0.0])
data_obs = []
data_actions = []

recording = False

print("\nControls:")
print("Use WASD/RFGT/YHUJIK for joint control")
print("Press ENTER to toggle recording ON/OFF")
print("Press ESC to save and exit\n")

for _ in range(999999):
    action, key = get_keyboard_action()
    if key == 27:  # ESC
        break
    if key == 13:  # ENTER toggles recording
        recording = not recording
        print("🔴 Recording ON" if recording else "⏸️ Recording OFF")

    obs, reward, done, info = venv.step([action])  # Vectorized env expects batched actions
    # Unbatch the returns
    obs = obs[0]
    reward = reward[0]
    done = done[0]
    if recording:
        data_obs.append(np.copy(obs))  # Copy after unbatching for consistent shape
        data_actions.append(action)

    # Render & debug
    frame = venv.render()
    if frame is not None:
        frame = np.array(frame)
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Get observation components

        components = simplify_obs(obs, include_velocities=True)  # obs is already unbatched

        # Calculate EE position
        ee_pos, ee_rot = franka.compute_ee_pose(components["joint_positions"][:7])
        roll, pitch, yaw = franka.rotation_to_rpy(ee_rot)

        # Display text
        rec_text = "RECORDING" if recording else "IDLE"
        color = (0, 0, 255) if recording else (128, 128, 128)

        # Text lines
        y_pos = 25
        line_height = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        cv2.putText(frame, f"{rec_text} | Steps: {len(data_obs)}", (10, y_pos), font, 0.7, color, 2)
        y_pos += 30

        cv2.putText(frame, f"Reward: {reward:.3f}", (10, y_pos), font, font_scale, (0, 255, 0), thickness)
        y_pos += line_height

        cv2.putText(
            frame,
            f"EE Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
            (10, y_pos),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
        )
        y_pos += line_height

        cv2.putText(
            frame,
            f"EE RPY: [{roll:.3f}, {pitch:.3f}, {yaw:.3f}]",
            (10, y_pos),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
        )
        y_pos += line_height

        # Joint positions
        joints_str = " ".join([f"{j:.2f}" for j in components["joint_positions"]])
        cv2.putText(frame, f"Joints: {joints_str}", (10, y_pos), font, font_scale, (255, 255, 0), thickness)
        y_pos += line_height

        # Ball, club, hole positions
        ball_pos = components["ball_pos"]
        club_pos = components["club_pos"]
        hole_pos = components["hole_pos"]

        cv2.putText(
            frame,
            f"Ball: [{ball_pos[0]:.3f}, {ball_pos[1]:.3f}, {ball_pos[2]:.3f}]",
            (10, y_pos),
            font,
            font_scale,
            (255, 255, 0),
            thickness,
        )
        y_pos += line_height

        cv2.putText(
            frame,
            f"Club: [{club_pos[0]:.3f}, {club_pos[1]:.3f}, {club_pos[2]:.3f}]",
            (10, y_pos),
            font,
            font_scale,
            (255, 255, 0),
            thickness,
        )
        y_pos += line_height

        cv2.putText(
            frame,
            f"Hole: [{hole_pos[0]:.3f}, {hole_pos[1]:.3f}, {hole_pos[2]:.3f}]",
            (10, y_pos),
            font,
            font_scale,
            (255, 255, 0),
            thickness,
        )
        y_pos += line_height

        # Distances (from info dict - true values, not normalized)
        ball_to_hole = info[0]["dist_ball_to_hole"]
        club_to_ball = info[0]["dist_club_to_ball"]
        ee_to_club = info[0]["dist_ee_to_club"]

        cv2.putText(frame, f"Ball->Hole: {ball_to_hole:.3f}", (10, y_pos), font, font_scale, (0, 255, 255), thickness)
        y_pos += line_height

        cv2.putText(frame, f"Club->Ball: {club_to_ball:.3f}", (10, y_pos), font, font_scale, (0, 255, 255), thickness)
        y_pos += line_height

        cv2.putText(frame, f"EE->Club: {ee_to_club:.3f}", (10, y_pos), font, font_scale, (0, 255, 255), thickness)

        cv2.imshow("Franka BC Recorder", frame)

    if done:
        obs = venv.reset()
        ee_pos = np.array([0.0, 0.0, 0.0])

venv.close()

# === Save dataset ===
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"expert_demo_{ts}.npz"
np.savez(save_path, observations=np.array(data_obs), actions=np.array(data_actions))
print(f"\n✅ Saved {len(data_obs)} steps to {save_path}")
