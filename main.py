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


# === DH parameters for Franka Panda ===
DH_PARAMS = [
    (0, 0.333, 0, None),  # Joint 1
    (0, 0, -np.pi / 2, None),  # Joint 2
    (0, 0.316, np.pi / 2, None),  # Joint 3
    (0.0825, 0, np.pi / 2, None),  # Joint 4
    (-0.0825, 0.384, -np.pi / 2, None),  # Joint 5
    (0, 0, np.pi / 2, None),  # Joint 6
    (0.088, 0, np.pi / 2, None),  # Joint 7
    (0, 0.107, 0, 0),  # Flange
    (0, 0.1034, 0, np.pi / 4),  # End effector
]


def dh_transform(a, d, alpha, theta):
    """Compute DH transformation matrix for given parameters."""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
            [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
            [0, 0, 0, 1],
        ]
    )


def compute_ee_position(joint_angles):
    """Compute end-effector (x,y,z) from 7 joint angles using Panda's DH parameters."""
    T = np.eye(4)  # pylint: disable=invalid-name
    for i, (a, d, alpha, theta) in enumerate(DH_PARAMS):
        if theta is None:
            theta = joint_angles[i]
        T = T @ dh_transform(a, d, alpha, theta)  # pylint: disable=invalid-name
    return T[:3, 3]  # Extract position vector (x, y, z)


def is_club_dropped(club_quat, club_pos, tilt_threshold_deg=60, height_threshold=0.05):
    """Detect if the club has dropped using orientation + height.

    Args:
        club_quat: Quaternion [w, x, y, z]
        club_pos: Club position [x, y, z]
        tilt_threshold_deg: Max allowed tilt angle from vertical
        height_threshold: Min allowed Z height before considering drop
    """
    w, x, y, z = club_quat
    # Rotation matrix from quaternion
    # pylint: disable=invalid-name
    R = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )

    z_axis = np.array([0, 0, 1])
    club_up = R[:, 2]  # Local Z-axis in world frame
    dot = np.dot(club_up, z_axis) / (np.linalg.norm(club_up) * np.linalg.norm(z_axis))
    tilt_angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    return tilt_angle > tilt_threshold_deg or club_pos[2] < height_threshold


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
        self.prev_ee_to_club = None  # Now tracked based on FK

    def reset(self, **kwargs):
        """Reset the environment and initialize distances."""
        obs, info = self.env.reset(**kwargs)

        # Extract positions
        ball_pos, hole_pos, club_pos = obs[18:21], obs[28:31], obs[21:24]
        ee_pos = compute_ee_position(obs[0:7])  # FK from 7 joint angles

        # Initialize distances
        self.prev_ball_to_hole = np.linalg.norm(ball_pos - hole_pos)
        self.prev_club_to_ball = np.linalg.norm(club_pos - ball_pos)
        self.prev_ee_to_club = np.linalg.norm(ee_pos - club_pos)

        return obs, info

    # pylint: disable=too-many-locals
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

        # Extract positions
        ball_pos, club_pos, hole_pos = obs[18:21], obs[21:24], obs[28:31]
        club_quat = obs[24:28]
        ee_pos = compute_ee_position(obs[0:7])  # Compute FK each step

        # Distances
        dist_ball_to_hole = np.linalg.norm(ball_pos - hole_pos)
        dist_club_to_ball = np.linalg.norm(club_pos - ball_pos)
        dist_ee_to_club = np.linalg.norm(ee_pos - club_pos)

        # Reward shaping
        shaped_reward = 0.0
        if dist_ee_to_club < self.prev_ee_to_club:
            shaped_reward += 0.5  # Reward for EE moving closer to club
        else:
            shaped_reward -= 0.1  # Penalty for moving away

        if dist_club_to_ball < self.prev_club_to_ball:
            shaped_reward += 0.5
        if dist_ball_to_hole < self.prev_ball_to_hole:
            shaped_reward += 1.0

        # Curriculum-based penalty for dropping the club
        if self.global_step < 500_000:
            drop_penalty_scale = 0.0
        elif self.global_step < 1_500_000:
            drop_penalty_scale = 0.5
        else:
            drop_penalty_scale = 1.0

        if is_club_dropped(club_quat, club_pos):
            shaped_reward += 2.0 * (1 - drop_penalty_scale)

        reward += shaped_reward

        # Update distances for next step
        self.prev_ball_to_hole = dist_ball_to_hole
        self.prev_club_to_ball = dist_club_to_ball
        self.prev_ee_to_club = dist_ee_to_club

        # Add extra info for logging
        info.update(
            {
                "dist_ball_to_hole": dist_ball_to_hole,
                "dist_club_to_ball": dist_club_to_ball,
                "dist_ee_to_club": dist_ee_to_club,
                "ee_position": ee_pos.tolist(),
                "shaped_reward": shaped_reward,
                "curriculum_scale": drop_penalty_scale,
            }
        )

        return obs, reward, terminated, truncated, info


# ===============================
# VIDEO RECORDING
# ===============================
# pylint: disable=too-many-locals
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
        obs, reward, done, _, _ = raw_env.step(action)
        frame = raw_env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Convert observations to a string to display
        chunks = [obs[0:10], obs[10:20], obs[20:31]]
        obs_text = "\n".join([" ".join([f"{val:.2f}" for val in chunk]) for chunk in chunks])

        reward_text = f"Reward: {reward:.2f}"

        # Add the observation and reward text to the frame
        cv2.putText(frame_bgr, obs_text, (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, reward_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

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
        self.base_video_dir = video_dir
        self.duration = duration
        self.display_live = display_live

        # Create a new folder (e.g., PPO1, PPO2, ...) based on existing directories
        run_number = len([d for d in os.listdir(self.base_video_dir) if d.startswith("PPO")]) + 1
        self.video_dir = os.path.join(self.base_video_dir, f"PPO{run_number}")
        os.makedirs(self.video_dir, exist_ok=True)  # Create the new run folder

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
            self.logger.record("custom/dist_ee_to_club", info["dist_ee_to_club"])
            self.logger.record("custom/ee_position", info["ee_position"])
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
policy_kwargs = {"net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]}, "activation_fn": torch.nn.ReLU}

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=lambda f: 1e-4 * f,
    n_steps=4096,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
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
