"""Reinforcement learning training script for Franka robot golf task with curriculum learning."""

import csv
import os

import gymnasium as gym
import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# -------- CONFIG --------
TOTAL_TIMESTEPS = 1_000_000
VIDEO_DIR = "./ppo_videos/"
LOG_DIR = "./ppo_logs/"
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CURRICULUM_STAGES = [
    {"difficulty": 0.1, "reward_bonus": 0.0},
    {"difficulty": 0.3, "reward_bonus": 0.2},
    {"difficulty": 0.6, "reward_bonus": 0.4},
    {"difficulty": 1.0, "reward_bonus": 0.6},
]


class CurriculumWrapper(gym.Wrapper):
    """Wrapper that implements curriculum learning by gradually increasing difficulty."""

    def __init__(self, env_instance, stages, steps_per_stage=250_000):
        super().__init__(env_instance)
        self.stages = stages
        self.steps_per_stage = steps_per_stage
        self.current_stage = 0
        self.total_steps = 0

    def reset(self, **kwargs):
        """Reset the environment and apply current difficulty level."""
        obs, info = self.env.reset(**kwargs)
        self._apply_difficulty()
        return obs, info

    def step(self, action):
        """Execute action, update curriculum stage, and apply reward bonus."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward += self.stages[self.current_stage]["reward_bonus"]

        self.total_steps += 1
        self._update_stage()  # Check every step for smooth curriculum

        return obs, reward, terminated, truncated, info

    def _update_stage(self):
        success = self.env.get_attr("success_rate")[0] if hasattr(self.env, "get_attr") else None
        if success and success > 0.7 and self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self._apply_difficulty()
            print(f"✅ Success {success:.2f}, advancing to stage {self.current_stage}")
        else:
            # Fallback to step-based if success metric is not available
            new_stage = min(self.total_steps // self.steps_per_stage, len(self.stages) - 1)
            if new_stage != self.current_stage:
                self.current_stage = new_stage
                self._apply_difficulty()
                print(
                    f"➡ Curriculum: Stage {self.current_stage} | "
                    f"Difficulty={self.stages[self.current_stage]['difficulty']}"
                )

    def _apply_difficulty(self):
        if hasattr(self.env, "set_difficulty"):
            self.env.set_difficulty(self.stages[self.current_stage]["difficulty"])


# -------- ENV FACTORY --------
def make_env():
    """Create and configure the environment with monitoring and curriculum learning."""
    sai = SAIClient(comp_id="franka-ml-hiring")
    env_instance = sai.make_env(render_mode="rgb_array")
    env_instance = Monitor(env_instance)
    return CurriculumWrapper(env_instance, CURRICULUM_STAGES)


# -------- TRAINING & EVAL ENVS --------
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=False)

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/", log_path="./logs/", eval_freq=10_000)

# -------- CUSTOM VIDEO CALLBACK --------


# pylint: disable=too-few-public-methods
class VideoRecorderCallback(BaseCallback):
    """Callback for recording videos of agent performance during training."""

    def __init__(self, video_freq, video_length, video_dir, verbose=0):
        super().__init__(verbose)
        self.video_freq = video_freq
        self.video_length = video_length
        self.video_dir = video_dir
        os.makedirs(video_dir, exist_ok=True)

    def _on_step(self) -> bool:
        """Check if it's time to record a video and trigger recording if needed."""
        if self.num_timesteps % self.video_freq == 0 and self.num_timesteps > 0:
            self.record_video()
        return True

    def record_video(self):
        """Record a video of the agent's current performance."""
        print(f"\nRecording video at step {self.num_timesteps}...")

        # Create evaluation environment
        video_env = DummyVecEnv([make_env])
        video_env = VecNormalize(video_env, training=False, norm_obs=True, norm_reward=False)

        # Sync normalization stats
        if hasattr(self.training_env, "obs_rms"):
            video_env.obs_rms = self.training_env.obs_rms

        frames = []
        obs = video_env.reset()
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, _ = video_env.step(action)

            frame = video_env.envs[0].render()  # Gymnasium default returns RGB array
            if frame is not None:
                frames.append(np.array(frame))
            if dones.any():
                break

        # Save video explicitly setting FPS
        if frames:
            # Import here to avoid loading the module when not needed
            # pylint: disable=import-outside-toplevel
            from moviepy.editor import ImageSequenceClip

            clip = ImageSequenceClip(frames, fps=30)
            clip.write_videofile(f"{self.video_dir}/step_{self.num_timesteps}-episode-0.mp4")
            print(f"Video saved: {self.video_dir}/step_{self.num_timesteps}-episode-0.mp4")
        else:
            print("No frames captured!")


# pylint: disable=too-few-public-methods
class DebugPolicyCallback(BaseCallback):
    """Callback for debugging policy behavior and detecting potential issues."""

    def __init__(self, debug_freq, n_steps, log_path="debug_policy_log.csv", verbose=1):
        super().__init__(verbose)
        self.debug_freq = debug_freq
        self.n_steps = n_steps
        self.log_path = log_path
        self.fields = ["step", "action", "reward", "delta_reward"]

        # Reset log file at start
        with open(self.log_path, "w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(self.fields)

    def _on_step(self) -> bool:
        """Check if it's time to run policy debugging and trigger if needed."""
        if self.num_timesteps % self.debug_freq == 0 and self.num_timesteps > 0:
            self.debug_policy()
        return True

    def debug_policy(self):
        """Run policy debugging with normal and sign-flipped actions."""
        print(f"\n[DEBUG] Policy check at step {self.num_timesteps}...")
        self._run_test(sign_flip=False, label="Normal")
        self._run_test(sign_flip=True, label="Flipped")

    def _run_test(self, sign_flip=False, label="Normal"):
        """Run a test rollout with optional action sign flipping for debugging."""
        print(f"\n--- {label} Rollout ---")
        debug_env = DummyVecEnv([make_env])
        debug_env = VecNormalize(debug_env, training=False, norm_obs=True, norm_reward=False)

        if hasattr(self.training_env, "obs_rms"):
            debug_env.obs_rms = self.training_env.obs_rms

        obs = debug_env.reset()
        prev_reward = 0
        rewards, delta_rewards, actions = [], [], []

        for step in range(self.n_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            if sign_flip:
                action = -action

            obs, reward, dones, _ = debug_env.step(action)

            delta = reward[0] - prev_reward
            prev_reward = reward[0]

            print(f"Step {step} | Action: {action} | Reward: {reward[0]:.4f} | " f"ΔReward: {delta:.4f}")
            rewards.append(reward[0])
            delta_rewards.append(delta)
            actions.append(action[0].tolist())

            if dones.any():
                obs = debug_env.reset()

        self._analyze_results(actions, delta_rewards, rewards, label)

    def _analyze_results(self, actions, delta_rewards, rewards, label):
        """Analyze test results to detect potential policy issues."""
        avg_reward = np.mean(rewards)
        avg_delta = np.mean(delta_rewards)

        print(f"\n[ANALYSIS] {label} Rollout:")
        print(f"  Avg Reward: {avg_reward:.4f}, Avg ΔReward: {avg_delta:.4f}")

        # Correlation check for sign reversal
        actions = np.array(actions)
        delta_rewards = np.array(delta_rewards)
        corr_flags = []
        for i in range(actions.shape[1]):
            corr = np.corrcoef(actions[:-1, i], delta_rewards[1:])[0, 1]
            status = "✅ OK" if corr < 0 else "⚠️ Possibly Reversed"
            # Add correlation and status to the list
            corr_flags.append((i, corr, status))

        print("\n[ANALYSIS] Correlation of actions vs ΔReward:")
        for i, corr, status in corr_flags:
            print(f"  Action[{i}] corr={corr:.3f} -> {status}")


video_callback = VideoRecorderCallback(video_freq=5_000, video_length=500, video_dir=VIDEO_DIR)
debug_callback = DebugPolicyCallback(debug_freq=5_000, n_steps=50)


policy_kwargs = {"net_arch": {"pi": [256, 256], "vf": [256, 256]}}
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=1e-4,  # Lower for stability
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.3,
    ent_coef=0.005,  # Encourage exploration
    verbose=1,
    tensorboard_log=LOG_DIR,
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_callback, video_callback, debug_callback])

# Save final model and normalization stats
model.save("franka_golf_curriculum")
env.save("vec_normalize.pkl")
env.close()
