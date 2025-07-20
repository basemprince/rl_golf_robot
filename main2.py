"""Video recording utility for Franka robot reinforcement learning training."""

import os

import cv2
import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sai = SAIClient("FrankaIkGolfCourseEnv-v0")
env = sai.make_env(render_mode="rgb_array")

env = DummyVecEnv([lambda: env])

os.makedirs("ppo_videos", exist_ok=True)


def record_video(env_to_record, filename, duration=15):
    """Record a video of the agent's performance.

    Args:
        env_to_record: Environment to record
        filename: Output video file path
        duration: Duration of recording in seconds
    """
    frame_width, frame_height = 1280, 720
    frame_rate = 30

    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

    obs = env_to_record.reset()  # Reset environment at the beginning of each episode
    done = False
    frames_recorded = 0

    # Use underscore for unused loop variable
    for _ in range(frame_rate * duration):
        if done:
            break
        action, _states = model.predict(obs)
        obs, _, done, _ = env_to_record.step(action)  # Ignore reward and info

        frame = env_to_record.render(mode="rgb_array")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        out.write(frame)
        frames_recorded += 1

        cv2.imshow("Training Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cv2.destroyAllWindows()


# pylint: disable=too-few-public-methods
class VideoRecordingCallback(BaseCallback):
    """Callback for recording videos at regular intervals during training."""

    def __init__(self, n_steps, training_env, video_dir="ppo_videos"):
        super().__init__()  # Use Python 3 style super()
        self.n_steps = n_steps
        self.training_env = training_env
        self.video_dir = video_dir
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1

        if self.step_count % self.n_steps == 0:
            filename = f"{self.video_dir}/video_{self.step_count}_steps.mp4"
            record_video(self.training_env, filename)
            print(f"Video for {self.step_count} steps saved as {filename}!")

        return True


video_callback = VideoRecordingCallback(n_steps=2000, training_env=env)

model = PPO("MlpPolicy", env, verbose=1)


def train_ppo_with_video(model_to_train, total_timesteps=10000, callback=None):
    """Train a PPO model with video recording.

    Args:
        model_to_train: PPO model to train
        total_timesteps: Total number of timesteps to train for
        callback: Optional callback for training
    """
    model_to_train.learn(total_timesteps=total_timesteps, callback=callback)


train_ppo_with_video(model, total_timesteps=100000, callback=video_callback)

env.close()
