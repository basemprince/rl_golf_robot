"""Video recording utility for Franka robot reinforcement learning training."""

import csv
import os

import gymnasium as gym
import numpy as np
from gymnasium.utils.save_video import save_video
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
                stage_msg = f"➡ Curriculum: Stage {self.current_stage}"
                diff_msg = f"Difficulty={self.stages[self.current_stage]['difficulty']}"
                print(f"{stage_msg} | {diff_msg}")

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

        # Initializing starting indices for steps and episodes
        self.step_starting_index = 0
        self.episode_index = 0

    def _on_step(self) -> bool:
        """Check if it's time to record a video and trigger recording if needed."""
        if self.num_timesteps % self.video_freq == 0 and self.num_timesteps > 0:
            self.record_video()
        return True

    def record_video(self):
        """Record a video of the agent's current performance using gymnasium's save_video."""
        print(f"\n🎥 Recording video at step {self.num_timesteps}...")

        local_env = make_env()  # Create a local environment for recording
        frames = []
        total_reward = 0

        # Reset the environment
        obs, _ = local_env.reset()

        # Extract the FPS from the environment's metadata
        fps = local_env.metadata.get("render_fps", 30)  # Default to 30 if no FPS is found in metadata
        if fps != 30:
            print(f"Using FPS from environment metadata: {fps}")
        else:
            print(f"Using default FPS: {fps}")

        # Run the episode and capture frames
        for _ in range(self.video_length):
            action, _ = self.model.predict(obs.reshape(1, -1), deterministic=True)
            obs, reward, terminated, truncated, _ = local_env.step(action)

            # Capture the frame
            frames.append(local_env.render())

            total_reward += reward

            # End the episode if done or truncated
            if terminated or truncated:
                break

        # Save the video after the episode ends, using the save_video function
        save_video(
            frames=frames,
            video_folder=self.video_dir,
            fps=fps,  # Use the FPS from metadata
            step_starting_index=self.step_starting_index,
            episode_index=self.episode_index,
        )

        print(f"✅ Video saved: {self.video_dir}/step_{self.num_timesteps}.mp4 (Reward: {total_reward:.2f})")

        # Update the indices for the next video
        self.step_starting_index = self.num_timesteps + 1
        self.episode_index += 1

        # Close the environment after recording
        local_env.close()


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

    # pylint: disable=too-many-locals, too-many-branches
    def debug_policy(self):
        """Run policy debugging with normal and sign-flipped actions."""
        print(f"\n[DEBUG] Policy check at step {self.num_timesteps}...")
        self._run_test(sign_flip=False, label="Normal")
        self._run_test(sign_flip=True, label="Flipped")

    def _run_test(self, sign_flip=False, label="Normal"):
        """Run a test rollout with optional action sign flipping for debugging."""
        print(f"\n--- {label} Rollout ---")
        # Create a vectorized environment for compatibility with the model
        debug_env = DummyVecEnv([make_env])
        debug_env = VecNormalize(debug_env, training=False, norm_obs=True, norm_reward=False)

        if hasattr(self.training_env, "obs_rms"):
            debug_env.obs_rms = self.training_env.obs_rms

        # Reset the environment - handle both return formats
        reset_result = debug_env.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, _ = reset_result
        else:
            obs = reset_result

        prev_reward = 0
        rewards, delta_rewards, actions = [], [], []

        for step in range(self.n_steps):
            # Get action from policy
            action, _ = self.model.predict(obs, deterministic=True)
            if sign_flip:
                action = -action

            # Execute action - handle both return formats
            step_result = debug_env.step(action)
            if len(step_result) == 5:  # New Gym API: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:  # Old Gym API: obs, reward, done, info
                obs, reward, done, _ = step_result
                terminated = done

            delta = reward[0] - prev_reward
            prev_reward = reward[0]

            print(f"Step {step} | Action: {action} | Reward: {reward[0]:.4f} | " f"ΔReward: {delta:.4f}")
            rewards.append(reward[0])
            delta_rewards.append(delta)
            actions.append(action[0].tolist())

            if isinstance(terminated, bool) and terminated:
                reset_result = debug_env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result
            elif hasattr(terminated, "any") and terminated.any():
                reset_result = debug_env.reset()
                if isinstance(reset_result, tuple):
                    obs, _ = reset_result
                else:
                    obs = reset_result

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


video_callback = VideoRecorderCallback(video_freq=2_000, video_length=500, video_dir=VIDEO_DIR)
debug_callback = DebugPolicyCallback(debug_freq=5_000, n_steps=50)


# Action space should match the environment's action space shape (7,)
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
