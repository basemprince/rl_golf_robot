"""Module for playing back behavioral cloning models."""

import time

import cv2
import numpy as np
from sai_rl import SAIClient
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.ppo import MlpPolicy

from main import FrankaFK, GolfRewardWrapper, simplify_obs

# Load the saved policy
policy = MlpPolicy.load("bc_policy_1.zip")

# Make the environment
sai = SAIClient("FrankaIkGolfCourseEnv-v0")
env = sai.make_env(render_mode="rgb_array")
franka = FrankaFK(env.spec)
env = GolfRewardWrapper(env, include_velocities=True, fk_solver=franka)

# Vectorize for compatibility
venv = DummyVecEnv([lambda: env])
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=5.0, clip_reward=10.0)
obs = venv.reset()
done = False

while not done:
    action, _ = policy.predict(obs, deterministic=True)
    print("Predicted action:", action)
    obs, reward, done, info = venv.step(action)

    # Optional rendering
    frame = venv.render()
    if frame is not None:
        frame = np.array(frame)
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Get observation components
        components = simplify_obs(obs[0], include_velocities=True)  # obs[0] to unbatch

        # Calculate EE position
        ee_pos, ee_rot = franka.compute_ee_pose(components["joint_positions"][:7])
        roll, pitch, yaw = franka.rotation_to_rpy(ee_rot)

        # Text lines
        y_pos = 25
        line_height = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        cv2.putText(frame, "BC Policy Rollout", (10, y_pos), font, 0.7, (255, 255, 255), 2)
        y_pos += 30

        cv2.putText(frame, f"Reward: {reward[0]:.3f}", (10, y_pos), font, font_scale, (0, 255, 0), thickness)
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

        cv2.imshow("BC Rollout", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    time.sleep(0.05)

venv.close()
