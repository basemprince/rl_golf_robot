"""
Reinforcement Learning for Franka Robot Golf Task using PPO.

This module implements a comprehensive training pipeline for the Franka robot golf task
using Proximal Policy Optimization (PPO) with several enhancements:

- Reward shaping via GolfRewardWrapper to guide learning
- Observation and reward normalization for training stability
- TensorBoard custom metrics logging for analysis
- Curriculum learning with gradually increasing drop penalties
- Entropy coefficient scheduling for balanced exploration
- Periodic video recording to visualize agent progress
- Evaluation callbacks to track and save best models

The environment simulates a Franka robot arm that must learn to use a golf club
to hit a ball into a hole, requiring complex manipulation skills.
"""

import os

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import NormalizeReward
from sai_rl import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ===============================
# CONFIG
# ===============================
VIDEO_DIR = "./ppo_videos"
LOG_DIR = "./ppo_logs"
BEST_MODEL_DIR = "./best_model"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 3_000_000
VIDEO_INTERVAL = 10_000
VIDEO_DURATION = 15
DISPLAY_LIVE = False


# ===============================
# GOLF REWARD WRAPPER
# ===============================
class GolfRewardWrapper(gym.Wrapper):
    """Wrapper that shapes rewards to guide learning in the golf environment.

    This wrapper enhances the sparse reward signal from the environment by adding:
    1. Rewards for moving the club closer to the ball
    2. Rewards for moving the ball closer to the hole
    3. Curriculum-based penalties for dropping the club

    It also tracks distances and provides them as additional info metrics.
    """

    # pylint: disable=redefined-outer-name
    def __init__(self, env):
        """Initialize the wrapper with the environment.

        Args:
            env: The environment to wrap
        """
        super().__init__(env)
        self.global_step = 0
        self.prev_ball_to_hole = None
        self.prev_club_to_ball = None

    def reset(self, **kwargs):
        """Reset the environment and initialize distance tracking.

        Args:
            **kwargs: Keyword arguments passed to the wrapped environment's reset method

        Returns:
            tuple: (observation, info)
        """
        obs, info = self.env.reset(**kwargs)
        ball_pos, hole_pos, club_pos = obs[18:21], obs[28:31], obs[21:24]
        self.prev_ball_to_hole = np.linalg.norm(ball_pos - hole_pos)
        self.prev_club_to_ball = np.linalg.norm(club_pos - ball_pos)
        return obs, info

    def step(self, action):
        """Execute action and apply reward shaping.

        This method enhances the original reward with shaped rewards based on:
        - Progress of the club toward the ball
        - Progress of the ball toward the hole
        - Curriculum-based penalties for dropping the club

        Args:
            action: The action to execute

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.global_step += 1

        ball_pos, club_pos, hole_pos = obs[18:21], obs[21:24], obs[28:31]
        dist_ball_to_hole = np.linalg.norm(ball_pos - hole_pos)
        dist_club_to_ball = np.linalg.norm(club_pos - ball_pos)

        shaped_reward = 0.0
        # Encourage club moving toward ball
        if dist_club_to_ball < self.prev_club_to_ball:
            shaped_reward += 0.5
        # Encourage ball progress toward hole
        if dist_ball_to_hole < self.prev_ball_to_hole:
            shaped_reward += 1.0

        # Curriculum penalty for dropping club
        if self.global_step < 500_000:
            drop_penalty_scale = 0.0
        elif self.global_step < 1_500_000:
            drop_penalty_scale = 0.5
        else:
            drop_penalty_scale = 1.0

        if "club_dropped" in info and info["club_dropped"]:
            shaped_reward -= 2.0 * drop_penalty_scale

        reward += shaped_reward

        self.prev_ball_to_hole = dist_ball_to_hole
        self.prev_club_to_ball = dist_club_to_ball

        info.update(
            {
                "dist_ball_to_hole": dist_ball_to_hole,
                "dist_club_to_ball": dist_club_to_ball,
                "shaped_reward": shaped_reward,
                "curriculum_scale": drop_penalty_scale,
            }
        )

        return obs, reward, terminated, truncated, info


# ===============================
# VIDEO RECORDING
# ===============================
def record_video(filename, duration=15, display_live=False):
    """Record a video of the agent's performance.

    Creates a video of the agent interacting with the environment using the current policy.
    The video is saved to the specified filename in mp4 format.

    Args:
        filename: Path where the video will be saved
        model: Trained model with a predict method
        duration: Duration of the video in seconds
        display_live: Whether to display the video in a window while recording
    """
    raw_env = make_raw_env()  # raw env without VecNormalize
    obs, _ = raw_env.reset()
    done = False
    frame_rate = 30

    frame = raw_env.render()  # Gymnasium now returns an RGB frame automatically
    h, w, _ = frame.shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (w, h))

    for _ in range(frame_rate * duration):
        if done:
            break
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = raw_env.step(action)
        frame = raw_env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        if display_live:
            cv2.imshow("Training Video", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    raw_env.close()
    out.release()
    if display_live:
        cv2.destroyAllWindows()


# ===============================
# CALLBACKS
# ===============================
# pylint: disable=too-many-positional-arguments, too-few-public-methods
class VideoRecordingCallback(BaseCallback):
    """Callback for recording videos at regular intervals during training.

    This callback periodically records videos of the agent's performance to visualize
    training progress and behavior changes over time.
    """

    # pylint: disable=redefined-outer-name
    def __init__(self, n_steps, model, video_dir=VIDEO_DIR, duration=15, display_live=False):
        """Initialize the callback.

        Args:
            n_steps: Number of steps between video recordings
            model: The model being trained
            video_dir: Directory where videos will be saved
            duration: Duration of each video in seconds
            display_live: Whether to display videos in a window while recording
        """

        super().__init__()
        self.n_steps = n_steps
        self.model = model
        self.video_dir = video_dir
        self.duration = duration
        self.display_live = display_live

    def _on_step(self):
        """Check if it's time to record a video and trigger recording if needed.

        Returns:
            bool: Whether to continue training (always True in this case)
        """
        if self.num_timesteps % self.n_steps == 0:
            filename = os.path.join(self.video_dir, f"video_{self.num_timesteps}_steps.mp4")
            record_video(filename, duration=self.duration, display_live=self.display_live)
            print(f"[INFO] Video saved at {filename}")
        return True


class EntropySchedulerCallback(BaseCallback):
    """Callback for scheduling entropy coefficient during training.

    This callback gradually decreases the entropy coefficient from an initial value
    to a final value over the course of training, balancing exploration and exploitation.
    """

    # pylint: disable=too-few-public-methods
    def __init__(self, initial_ent=0.05, final_ent=0.005, total_steps=TOTAL_TIMESTEPS):
        """Initialize the callback.

        Args:
            initial_ent: Initial entropy coefficient value
            final_ent: Final entropy coefficient value
            total_steps: Total number of training steps
        """
        super().__init__()
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.total_steps = total_steps

    def _on_step(self):
        """Update the entropy coefficient based on training progress.

        Linearly interpolates between initial and final entropy values based on
        the fraction of training completed.

        Returns:
            bool: Whether to continue training (always True in this case)
        """
        frac = self.num_timesteps / self.total_steps
        new_ent = self.initial_ent + frac * (self.final_ent - self.initial_ent)
        self.model.ent_coef = max(self.final_ent, new_ent)
        return True


class TensorboardMetricsCallback(BaseCallback):
    """Callback for logging custom metrics to TensorBoard.

    This callback extracts custom metrics from the environment info dictionary
    and logs them to TensorBoard for visualization and analysis.
    """

    def _on_step(self):
        """Log custom metrics to TensorBoard.

        Extracts metrics from the environment info dictionary and logs them
        to TensorBoard using the logger.

        Returns:
            bool: Whether to continue training (always True in this case)
        """
        info = self.locals.get("infos")[0]
        if "dist_ball_to_hole" in info:
            self.logger.record("custom/dist_ball_to_hole", info["dist_ball_to_hole"])
            self.logger.record("custom/dist_club_to_ball", info["dist_club_to_ball"])
            self.logger.record("custom/shaped_reward", info["shaped_reward"])
            self.logger.record("custom/curriculum_scale", info["curriculum_scale"])
        return True


# ===============================
# ENVIRONMENT SETUP
# ===============================
sai = SAIClient("FrankaIkGolfCourseEnv-v0")

# Training environment (NO rendering for speed)
train_env = sai.make_env()
train_env = GolfRewardWrapper(train_env)
train_env = NormalizeReward(train_env, gamma=0.99)

# Wrap training env with VecEnv + VecNormalize
env = DummyVecEnv([lambda: train_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Evaluation environment
eval_env = DummyVecEnv([lambda: sai.make_env()])  # pylint: disable=unnecessary-lambda
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# ===============================
# PPO CONFIG
# ===============================
policy_kwargs = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "activation_fn": torch.nn.ReLU}

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=lambda f: 3e-4 * f,
    n_steps=2048,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log=LOG_DIR,
)


def make_raw_env():
    """Create a raw environment for video recording.

    Creates an environment with rendering enabled and reward shaping applied,
    but without vectorization or normalization, suitable for video recording.

    Returns:
        gym.Env: A raw environment instance with rendering enabled
    """
    raw_env = sai.make_env(render_mode="rgb_array")
    raw_env = GolfRewardWrapper(raw_env)  # still apply shaping for consistency
    return raw_env


# ===============================
# TRAIN
# ===============================
def train(total_timesteps=TOTAL_TIMESTEPS, display_live_video=False):
    """Train the PPO agent on the Franka golf task.

    Sets up all callbacks and runs the training process for the specified number of timesteps.

    Args:
        total_timesteps: Total number of environment steps to train for
        display_live_video: Whether to display videos in a window during recording
    """
    video_callback = VideoRecordingCallback(
        VIDEO_INTERVAL, model, duration=VIDEO_DURATION, display_live=display_live_video
    )
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=BEST_MODEL_DIR, log_path="./eval_logs", eval_freq=50_000
    )
    entropy_callback = EntropySchedulerCallback()
    tb_callback = TensorboardMetricsCallback()

    model.learn(
        total_timesteps=total_timesteps, callback=[video_callback, eval_callback, entropy_callback, tb_callback]
    )
    print("[INFO] Training completed!")


if __name__ == "__main__":
    train(display_live_video=DISPLAY_LIVE)
    env.close()
    eval_env.close()
