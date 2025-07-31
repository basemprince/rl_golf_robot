#!/usr/bin/env python3
"""
Play a trained model and record videos of its performance.

Usage:
    python play_model.py <model_path>

Example:
    python play_model.py ./experiments/best_experiment/final_model.zip
"""

import os
import sys

import cv2
import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from main import FrankaFK, GolfRewardWrapper, SimplifiedObservationWrapper, step_and_render_env


# pylint: disable=too-many-locals, redefined-outer-name
def play_model(model_path, bc_policy_path, num_episodes=10, video_duration=15, include_velocities=False):
    """Play a trained model live.

    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to play
        video_duration: Duration of each episode in seconds
        include_velocities: Whether the model was trained with velocities
    """
    if bc_policy_path is not None:
        model = MlpPolicy.load(bc_policy_path)
    # Load the model
    else:
        model = PPO.load(model_path)

    # Create environment for live display
    sai = SAIClient("FrankaIkGolfCourseEnv-v0")
    raw_env = sai.make_env(render_mode="rgb_array")
    fk_solver = FrankaFK(raw_env.spec)
    raw_env = GolfRewardWrapper(raw_env, include_velocities=include_velocities, fk_solver=fk_solver)
    raw_env = SimplifiedObservationWrapper(raw_env, include_velocities=include_velocities)

    frame_rate = 30

    for episode in range(num_episodes):
        print(f"Playing episode {episode + 1}/{num_episodes} (Press 'q' to quit, 'n' for next episode)")

        obs, _ = raw_env.reset()
        done = False

        frame_count = 0
        max_frames = frame_rate * video_duration

        while not done and frame_count < max_frames:
            # Use helper function to step and render
            obs, reward, done, frame_bgr, components = step_and_render_env(model, obs, raw_env, include_velocities)

            # Add text overlay with episode info
            ball_pos = components["ball_pos"]
            club_pos = components["club_pos"]
            hole_pos = components["hole_pos"]

            # Calculate distances
            ball_to_hole = np.linalg.norm(ball_pos - hole_pos)
            club_to_ball = np.linalg.norm(club_pos - ball_pos)

            # Add text
            cv2.putText(
                frame_bgr,
                f"Episode: {episode + 1}/{num_episodes}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(frame_bgr, f"Reward: {reward:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(
                frame_bgr, f"Ball-to-hole: {ball_to_hole:.3f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            cv2.putText(
                frame_bgr, f"Club-to-ball: {club_to_ball:.3f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

            # Display frame
            cv2.imshow("Model Performance", frame_bgr)

            # Handle key presses
            key = cv2.waitKey(33) & 0xFF  # ~30 FPS
            if key == ord("q"):
                raw_env.close()
                cv2.destroyAllWindows()
                return
            if key == ord("n"):
                break

            frame_count += 1

    raw_env.close()
    cv2.destroyAllWindows()
    print("Finished playing all episodes")


if __name__ == "__main__":

    model_path = None
    bc_policy_path = None
    args = sys.argv[1:]
    if "-p" in args:
        idx = args.index("-p")
        model_path = args[idx + 1]
    if "-b" in args:
        idx = args.index("-b")
        bc_policy_path = args[idx + 1]

    if model_path and not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Try to determine if model was trained with velocities by checking config
    # config_path = os.path.join(os.path.dirname(model_path), "config.json")
    INCLUDE_VELOCITIES = True

    # if os.path.exists(config_path):
    #     import json

    #     with open(config_path, "r", encoding="utf-8") as f:
    #         config = json.load(f)
    #         INCLUDE_VELOCITIES = config.get("include_velocities", False)
    #     print(f"Found config file. Using include_velocities={INCLUDE_VELOCITIES}")
    # else:
    #     print("No config file found. Assuming include_velocities=False")

    # print("Controls: Press 'q' to quit, 'n' to skip to next episode")

    play_model(model_path, bc_policy_path, num_episodes=10, include_velocities=INCLUDE_VELOCITIES)
