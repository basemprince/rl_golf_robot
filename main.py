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
from sai_rl import SAIClient
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import A2C
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
DISPLAY_LIVE = True
INCLUDE_VELOCITIES = False


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
    rot_matrix = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )

    z_axis = np.array([0, 0, 1])
    club_up = rot_matrix[:, 2]  # Local Z-axis in world frame
    dot = np.dot(club_up, z_axis) / (np.linalg.norm(club_up) * np.linalg.norm(z_axis))
    tilt_angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    return tilt_angle > tilt_threshold_deg or club_pos[2] < height_threshold


class FrankaFK:
    """Forward Kinematics for Franka Panda in world frame using DH parameters."""

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

    def __init__(self, env_spec):
        """Initialize FK solver using env.spec."""
        config = env_spec.kwargs.get("env_config", {})
        self.base_pos = np.array(config.get("robot_pos", [0, 0, 0]), dtype=np.float32)
        quat = config.get("robot_quat", [1, 0, 0, 0])  # [w, x, y, z]
        # Convert to scipy Rotation (expects [x, y, z, w])
        self.base_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])

    @staticmethod
    def dh_transform(a, d, alpha, theta):
        """Compute DH transformation matrix."""
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0, a],
                [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
                [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
                [0, 0, 0, 1],
            ]
        )

    def compute_ee_position(self, joint_angles):
        """Compute end-effector position in world frame."""
        # FK in robot base frame
        # pylint: disable=invalid-name
        T = np.eye(4)
        for i, (a, d, alpha, theta) in enumerate(self.DH_PARAMS):
            if theta is None:
                theta = joint_angles[i]
            T = T @ self.dh_transform(a, d, alpha, theta)  # pylint: disable=invalid-name
        ee_in_base = T[:3, 3]

        # Transform to world frame
        return self.base_rot.apply(ee_in_base) + self.base_pos


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
    def __init__(self, env, include_velocities=False, fk_solver=None):
        """Initialize the wrapper with the environment.

        Args:
            env: The environment to wrap
            include_velocities: Whether joint velocities are included in observations
        """
        super().__init__(env)
        self.global_step = 0
        self.prev_ball_to_hole = None
        self.prev_club_to_ball = None
        self.prev_ee_to_club = None  # Now tracked based on FK
        self.include_velocities = include_velocities
        self.fk_solver = fk_solver

    def reset(self, **kwargs):
        """Reset the environment and initialize distances."""
        obs, info = self.env.reset(**kwargs)

        # Get simplified observation with named components
        components = simplify_obs(obs, self.include_velocities)

        # Extract positions directly from the dictionary
        ball_pos = components["ball_pos"]
        club_pos = components["club_pos"]
        hole_pos = components["hole_pos"]

        ee_pos = self.fk_solver.compute_ee_position(obs[0:7])  # FK from 7 joint angles

        # Initialize distances
        self.prev_ball_to_hole = np.linalg.norm(ball_pos - hole_pos)
        self.prev_club_to_ball = np.linalg.norm(club_pos - ball_pos)
        self.prev_ee_to_club = np.linalg.norm(ee_pos - club_pos)

        return obs, info

    # pylint: disable=too-many-locals
    def step(self, action):
        """Execute action and apply improved reward shaping.

        Shaped rewards now include:
        - Progress-based rewards (positive for improvement, negative for regress)
        - Distance-based attraction terms
        - Club→ball and ball→hole incentives
        - Curriculum-based penalty for dropping the club
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.global_step += 1

        # Get simplified observation with named components
        components = simplify_obs(obs, self.include_velocities)

        # Extract positions directly from the dictionary
        ball_pos = components["ball_pos"]
        club_pos = components["club_pos"]
        club_quat = components["club_quat"]
        hole_pos = components["hole_pos"]

        ee_pos = self.fk_solver.compute_ee_position(obs[0:7])  # Forward kinematics for EE

        # Compute distances
        dist_ball_to_hole = np.linalg.norm(ball_pos - hole_pos)
        dist_club_to_ball = np.linalg.norm(club_pos - ball_pos)
        dist_ee_to_club = np.linalg.norm(ee_pos - club_pos)

        # Compute progress terms
        progress_ee_to_club = self.prev_ee_to_club - dist_ee_to_club
        progress_club_to_ball = self.prev_club_to_ball - dist_club_to_ball
        progress_ball_to_hole = self.prev_ball_to_hole - dist_ball_to_hole

        # Initialize shaped reward
        shaped_reward = 0.0

        # =========================
        # 1. EE → Club shaping
        # =========================
        shaped_reward += 20 * progress_ee_to_club  # Positive for improvement, negative for regress
        shaped_reward += 2.0 / (dist_ee_to_club + 1e-6)  # Attraction term for staying close

        # =========================
        # 2. Club → Ball shaping
        # =========================
        shaped_reward += 2 * progress_club_to_ball
        shaped_reward += 1.0 / (dist_club_to_ball + 1e-6)

        # =========================
        # 3. Ball → Hole shaping
        # =========================
        shaped_reward += 3 * progress_ball_to_hole
        shaped_reward += 1.0 / (dist_ball_to_hole + 1e-6)

        # =========================
        # 4. Penalty for dropped club
        # =========================
        if self.global_step < 500_000:
            drop_penalty_scale = 0.0
        elif self.global_step < 1_500_000:
            drop_penalty_scale = 0.5
        else:
            drop_penalty_scale = 1.0

        min_drop_height = 0.25
        if is_club_dropped(club_quat, club_pos, height_threshold=min_drop_height):  # Custom check based on quaternion
            shaped_reward += 2.0 * (1 - drop_penalty_scale)

        # Final reward is shaped reward (we override sparse reward)
        old_reward = reward
        reward += shaped_reward

        # Update previous distances
        self.prev_ball_to_hole = dist_ball_to_hole
        self.prev_club_to_ball = dist_club_to_ball
        self.prev_ee_to_club = dist_ee_to_club

        # Info for logging
        info.update(
            {
                "dist_ball_to_hole": dist_ball_to_hole,
                "dist_club_to_ball": dist_club_to_ball,
                "dist_ee_to_club": dist_ee_to_club,
                "progress_ee_to_club": progress_ee_to_club,
                "progress_club_to_ball": progress_club_to_ball,
                "progress_ball_to_hole": progress_ball_to_hole,
                "shaped_reward": shaped_reward,
                "old_reward": old_reward,
                "curriculum_scale": drop_penalty_scale,
            }
        )

        return obs, reward, terminated, truncated, info


# pylint: disable=too-few-public-methods
class SimplifiedObservationWrapper(gym.ObservationWrapper):
    """
    Reduces observation to only essential elements:
    - 7 joint positions (+ 2 gripper joints)
    - Optional joint velocities (or zeros if not included)
    - Ball position (x, y, z)
    - Club position (x, y, z)
    - Club orientation (w, x, y, z)
    - Hole position (x, y, z)
    """

    # pylint: disable=redefined-outer-name
    def __init__(self, env, include_velocities=False):
        super().__init__(env)
        self.include_velocities = include_velocities
        # Always use 31 elements for the observation space to match the model's expectations
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)

    def observation(self, obs):
        """Simplify the observation to match the model's input."""
        return simplify_obs(obs, self.include_velocities)["array"]


def simplify_obs(obs, include_velocities=False):
    """Simplify the observation to match the model's input.

    Args:
        obs: Raw observation from the environment
        include_velocities: Whether to include joint velocities in the simplified observation

    Returns:
        Dictionary containing observation components and the full simplified observation array
    """
    joint_positions = obs[0:9]  # 9 values: 7 joints + 2 gripper
    result = {"joint_positions": joint_positions}

    if include_velocities:
        # With velocities - use all 31 elements
        joint_velocities = obs[9:18]  # Joint velocities
        ball_pos = obs[18:21]
        club_pos = obs[21:24]
        club_quat = obs[24:28]
        hole_pos = obs[28:31]
        result["joint_velocities"] = joint_velocities
    else:
        # Without velocities - need to pad to match the expected 31 elements
        if len(obs) == 31:  # Full observation with velocities
            ball_pos = obs[18:21]
            club_pos = obs[21:24]
            club_quat = obs[24:28]
            hole_pos = obs[28:31]
        else:  # Already simplified observation without velocities
            ball_pos = obs[9:12]  # Shifted from 18:21 to 9:12
            club_pos = obs[12:15]  # Shifted from 21:24 to 12:15
            club_quat = obs[15:19]  # Shifted from 24:28 to 15:19
            hole_pos = obs[19:22]  # Shifted from 28:31 to 19:22

        # Create zero-filled array for velocities
        zero_velocities = np.zeros(9, dtype=np.float32)  # 9 zeros for velocities
        result["joint_velocities"] = zero_velocities

    # Store all components in the dictionary
    result["ball_pos"] = ball_pos
    result["club_pos"] = club_pos
    result["club_quat"] = club_quat
    result["hole_pos"] = hole_pos

    # Also create the full simplified observation array for the model
    simplified_array = np.concatenate(
        [joint_positions, result["joint_velocities"], ball_pos, club_pos, club_quat, hole_pos]
    ).astype(np.float32)
    result["array"] = simplified_array

    return result


# ===============================
# VIDEO RECORDING
# ===============================
# pylint: disable=too-many-locals, redefined-outer-name
def record_video(
    filename, duration=15, display_live=False, include_velocities=False, fk_solver=None
):  # pylint: disable=missing-function-docstring
    """Record a video of the agent's performance.

    Creates a video of the agent interacting with the environment using the current policy.
    The video is saved to the specified filename in mp4 format.

    Args:
        filename: Path where the video will be saved
        model: Trained model with a predict method
        duration: Duration of the video in seconds
        display_live: Whether to display the video in a window while recording
    """
    raw_env = make_raw_env(include_velocities=include_velocities, fk_solver=fk_solver)  # raw env without VecNormalize
    obs, _ = raw_env.reset()
    done = False
    frame_rate = 30

    frame = raw_env.render()  # Gymnasium now returns an RGB frame automatically
    h, w, _ = frame.shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (w, h))

    for _ in range(frame_rate * duration):
        if done:
            break
        # Always use the raw observation for prediction to ensure consistent dimensions
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = raw_env.step(action)
        frame = raw_env.render()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Get simplified observation with named components
        components = simplify_obs(obs, include_velocities)

        # Extract components directly from the dictionary
        joint_pos = components["joint_positions"]  # 7 DOF + 2 gripper joints
        ball_pos = components["ball_pos"]  # (x, y, z)
        club_pos = components["club_pos"]  # (x, y, z)
        club_quat = components["club_quat"]  # (w, x, y, z)
        hole_pos = components["hole_pos"]  # (x, y, z)

        ee_pos = fk_solver.compute_ee_position(obs[0:7])  # First 7 are joint angles
        ee_text = f"EE Pos: {ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}"

        # Format text with labels
        obs_text_lines = [
            f"Joint Pos: {' '.join([f'{val:.2f}' for val in joint_pos])}",
        ]

        # Only include velocities if the toggle is on
        if include_velocities:
            joint_vel = components["joint_velocities"]  # 7 DOF + 2 gripper joints
            obs_text_lines.append(f"Joint Vel: {' '.join([f'{val:.2f}' for val in joint_vel])}")

        # Add the rest of the observation components
        obs_text_lines.extend(
            [
                f"Ball Pos:  {' '.join([f'{val:.2f}' for val in ball_pos])}",
                f"Club Pos:  {' '.join([f'{val:.2f}' for val in club_pos])}",
                f"Club Ori:  {' '.join([f'{val:.2f}' for val in club_quat])}",
                f"Hole Pos:  {' '.join([f'{val:.2f}' for val in hole_pos])}",
                ee_text,
            ]
        )

        reward_text = f"Reward: {reward:.2f}"

        # Add observation lines to frame
        y_offset = h - 140
        for line in obs_text_lines:
            cv2.putText(frame_bgr, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            y_offset += 15

        cv2.putText(frame_bgr, reward_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

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
# pylint: disable=too-many-positional-arguments, too-few-public-methods, too-many-instance-attributes
class VideoRecordingCallback(BaseCallback):
    """Callback for recording videos at regular intervals during training.

    This callback periodically records videos of the agent's performance to visualize
    training progress and behavior changes over time.
    """

    # pylint: disable=redefined-outer-name, too-many-arguments
    def __init__(
        self,
        n_steps,
        model,
        video_dir=VIDEO_DIR,
        duration=15,
        display_live=False,
        include_velocities=False,
        fk_solver=None,
    ):
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
        self.include_velocities = include_velocities
        self.fk_solver = fk_solver

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
            record_video(
                filename,
                duration=self.duration,
                display_live=self.display_live,
                include_velocities=self.include_velocities,
                fk_solver=self.fk_solver,
            )
            print(f"[INFO] Video saved at {filename}")
        return True


class EntropySchedulerCallback(BaseCallback):  # pylint: disable=too-few-public-methods
    """Callback for scheduling entropy coefficient during training.

    This callback gradually decreases the entropy coefficient from an initial value
    to a final value over the course of training, balancing exploration and exploitation.
    """

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
            self.logger.record("custom/progress_ee_to_club", info["progress_ee_to_club"])
            self.logger.record("custom/progress_club_to_ball", info["progress_club_to_ball"])
            self.logger.record("custom/progress_ball_to_hole", info["progress_ball_to_hole"])
            self.logger.record("custom/shaped_reward", info["shaped_reward"])
            self.logger.record("custom/old_reward", info["old_reward"])
            self.logger.record("custom/curriculum_scale", info["curriculum_scale"])
        return True


# ===============================
# ENVIRONMENT SETUP
# ===============================
sai = SAIClient(comp_id="franka-ml-hiring")

# Toggle for including velocities in observations


# Training environment (NO rendering for speed)
train_env = sai.make_env()  # Full env
fk_solver = FrankaFK(train_env.spec)
train_env = GolfRewardWrapper(train_env, include_velocities=INCLUDE_VELOCITIES, fk_solver=fk_solver)  # Reward shaping
train_env = SimplifiedObservationWrapper(train_env, include_velocities=INCLUDE_VELOCITIES)  # Reduce obs size
env = DummyVecEnv([lambda: train_env])  # Vectorize
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=3.0, clip_reward=10.0)


# Evaluation environment
eval_env = DummyVecEnv(
    [
        lambda: SimplifiedObservationWrapper(
            GolfRewardWrapper(sai.make_env(), include_velocities=INCLUDE_VELOCITIES, fk_solver=fk_solver),
            include_velocities=INCLUDE_VELOCITIES,
        )
    ]
)  # pylint: disable=unnecessary-lambda
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=3.0, clip_reward=10.0)

# ===============================
# PPO CONFIG
# ===============================
# Define the observation dimension based on whether velocities are included
OBS_DIM = 31  # Always use 31 dimensions for consistency
policy_kwargs = {"net_arch": {"pi": [256, 256, 128], "vf": [256, 256, 128]}, "activation_fn": torch.nn.ReLU}

model = A2C(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=lambda f: 3e-5 * f,
    n_steps=4096,  # Longer rollouts
    # batch_size=256,  # Keep mini-batch moderate
    gamma=0.99,
    gae_lambda=0.95,
    # clip_range=0.2,
    ent_coef=0.02,  # Slightly more exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log=LOG_DIR,
)


def make_raw_env(include_velocities=INCLUDE_VELOCITIES, fk_solver=None):  # pylint: disable=redefined-outer-name
    """Create a raw environment for video recording.

    Creates an environment with rendering enabled and reward shaping applied,
    but without vectorization or normalization, suitable for video recording.

    Args:
        include_velocities: Whether to include joint velocities in observations

    Returns:
        gym.Env: A raw environment instance with rendering enabled
    """
    raw_env = sai.make_env(render_mode="rgb_array")
    # Create a new FK solver if one wasn't provided
    if fk_solver is None:
        fk_solver = FrankaFK(raw_env.spec)
    raw_env = GolfRewardWrapper(raw_env, include_velocities=include_velocities, fk_solver=fk_solver)
    raw_env = SimplifiedObservationWrapper(raw_env, include_velocities=include_velocities)
    return raw_env


# ===============================
# TRAIN
# ===============================
# pylint: disable=redefined-outer-name
def train(
    total_timesteps=TOTAL_TIMESTEPS, display_live_video=False, include_velocities=INCLUDE_VELOCITIES, fk_solver=None
):
    """Train the PPO agent on the Franka golf task.

    Sets up all callbacks and runs the training process for the specified number of timesteps.

    Args:
        total_timesteps: Total number of environment steps to train for
        display_live_video: Whether to display videos in a window during recording
        include_velocities: Whether to include joint velocities in observations
    """

    video_callback = VideoRecordingCallback(
        VIDEO_INTERVAL,
        model,
        duration=VIDEO_DURATION,
        display_live=display_live_video,
        include_velocities=include_velocities,
        fk_solver=fk_solver,
    )
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=BEST_MODEL_DIR, log_path="./eval_logs", eval_freq=50_000
    )
    entropy_callback = EntropySchedulerCallback()
    tb_callback = TensorboardMetricsCallback()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[video_callback, eval_callback, entropy_callback, tb_callback],
        progress_bar=True,
    )
    print("[INFO] Training completed!")


if __name__ == "__main__":
    train(display_live_video=DISPLAY_LIVE, include_velocities=INCLUDE_VELOCITIES, fk_solver=fk_solver)
    env.close()
    eval_env.close()
