# pylint: disable=duplicate-code
"""
Hyperparameter Experimentation Framework for RL Golf Robot.

This module provides tools to run multiple experiments with different hyperparameter
configurations and track their performance metrics.
"""

import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from main import (
    DummyVecEnv,
    EntropySchedulerCallback,
    FrankaFK,
    GolfRewardWrapper,
    SimplifiedObservationWrapper,
    TensorboardMetricsCallback,
    sai,
)


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# Directory setup
EXPERIMENTS_DIR = "./experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


# pylint: disable=too-few-public-methods
class MetricTrackingCallback(BaseCallback):
    """Callback for tracking and storing metrics during training.

    This callback collects key performance metrics at regular intervals
    and stores them for later analysis and comparison.
    """

    def __init__(self, eval_env, eval_freq=5000, verbose=0):
        """Initialize the callback.

        Args:
            eval_env: Environment used for evaluation
            eval_freq: Frequency of evaluation in timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.metrics: dict = {
            "timesteps": [],
            "mean_reward": [],
            "std_reward": [],
            "mean_dist_ball_to_hole": [],
            "mean_dist_club_to_ball": [],
            "success_rate": [],
        }

    def _on_step(self):
        """Evaluate the agent and record metrics at regular intervals."""
        if self.num_timesteps % self.eval_freq == 0:
            metrics = self._run_evaluation()
            self._store_metrics(metrics)
            if self.verbose > 0:
                self._log_metrics(metrics)
        return True

    def _run_evaluation(self):
        """Run evaluation episodes and collect metrics."""
        episode_rewards = []
        episode_ball_to_hole = []
        episode_club_to_ball = []
        success_count = 0
        n_eval_episodes = 5

        for _ in range(n_eval_episodes):
            episode_data = self._run_single_episode()
            episode_rewards.append(episode_data["reward"])

            if episode_data["ball_to_hole"] is not None:
                episode_ball_to_hole.append(episode_data["ball_to_hole"])
                if episode_data["ball_to_hole"] < 0.1:
                    success_count += 1

            if episode_data["club_to_ball"] is not None:
                episode_club_to_ball.append(episode_data["club_to_ball"])

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_ball_to_hole": np.mean(episode_ball_to_hole) if episode_ball_to_hole else float("nan"),
            "mean_club_to_ball": np.mean(episode_club_to_ball) if episode_club_to_ball else float("nan"),
            "success_rate": success_count / n_eval_episodes,
        }

    def _run_single_episode(self):
        """Run a single evaluation episode."""
        obs = self.eval_env.reset()[0]
        done = False
        episode_reward = 0
        final_ball_to_hole = None
        final_club_to_ball = None

        while not done:
            action = self._get_action(obs)
            obs, reward, done, info = self._step_environment(action)
            episode_reward += reward

            # Extract metrics from info
            info_dict = info[0] if isinstance(info, list) and len(info) > 0 else info
            if "dist_ball_to_hole" in info_dict:
                final_ball_to_hole = info_dict["dist_ball_to_hole"]
                final_club_to_ball = info_dict["dist_club_to_ball"]

        return {"reward": episode_reward, "ball_to_hole": final_ball_to_hole, "club_to_ball": final_club_to_ball}

    def _get_action(self, obs):
        """Get and format action from the model."""
        action, _ = self.model.predict(obs, deterministic=True)

        # Ensure action is properly formatted
        if not isinstance(action, np.ndarray) or action.size != 7:
            action = self._format_action(action)

        # Ensure action is 2D for vectorized environments
        if action.ndim == 1:
            action = action.reshape(1, -1)

        return action

    def _format_action(self, action):
        """Format action to correct shape and type."""
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                return np.full(7, float(action))
            if action.size == 1:
                return np.full(7, float(action.item()))
            return np.array([0.0] * 7)
        return np.array([0.0] * 7)

    def _step_environment(self, action):
        """Step the environment and handle different API versions."""
        step_result = self.eval_env.step(action)
        if len(step_result) == 5:  # New Gym API
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:  # Old Gym API
            obs, reward, done, info = step_result
        return obs, reward, done, info

    def _store_metrics(self, metrics):
        """Store metrics in the tracking dictionary."""
        self.metrics["timesteps"].append(self.num_timesteps)
        self.metrics["mean_reward"].append(metrics["mean_reward"])
        self.metrics["std_reward"].append(metrics["std_reward"])
        self.metrics["mean_dist_ball_to_hole"].append(metrics["mean_ball_to_hole"])
        self.metrics["mean_dist_club_to_ball"].append(metrics["mean_club_to_ball"])
        self.metrics["success_rate"].append(metrics["success_rate"])

    def _log_metrics(self, metrics):
        """Log metrics to console."""
        print(f"Timestep: {self.num_timesteps}")
        print(f"Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Mean ball-to-hole distance: {metrics['mean_ball_to_hole']:.4f}")
        print(f"Success rate: {metrics['success_rate']:.2f}")
        print("-" * 50)


class ExperimentManager:
    """Manages multiple experiments with different hyperparameter configurations."""

    def __init__(self, base_dir=EXPERIMENTS_DIR):
        """Initialize the experiment manager.

        Args:
            base_dir: Base directory for storing experiment results
        """
        self.base_dir = base_dir
        self.results = {}

    # pylint: disable=too-many-locals
    def run_experiment(self, config, experiment_name=None, total_timesteps=1_000_000):
        """Run a single experiment with the given configuration.

        Args:
            config: Dictionary containing hyperparameter configuration
            experiment_name: Name for this experiment (auto-generated if None)
            total_timesteps: Total timesteps to train for

        Returns:
            Dictionary containing experiment results and metrics
        """
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"

        # Create experiment directory
        exp_dir = os.path.join(self.base_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Save configuration
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Setup environment
        train_env = sai.make_env()
        fk_solver = FrankaFK(train_env.spec)

        include_velocities = config.get("include_velocities", False)

        # Apply wrappers
        train_env = GolfRewardWrapper(train_env, include_velocities=include_velocities, fk_solver=fk_solver)
        train_env = SimplifiedObservationWrapper(train_env, include_velocities=include_velocities)

        # Vectorize and normalize
        env = DummyVecEnv([lambda: train_env])
        env = VecNormalize(
            env,
            norm_obs=config.get("norm_obs", True),
            norm_reward=config.get("norm_reward", True),
            clip_obs=config.get("clip_obs", 3.0),
            clip_reward=config.get("clip_reward", 10.0),
        )

        # Setup evaluation environment
        eval_env = DummyVecEnv(
            [
                lambda: SimplifiedObservationWrapper(
                    GolfRewardWrapper(sai.make_env(), include_velocities=include_velocities, fk_solver=fk_solver),
                    include_velocities=include_velocities,
                )
            ]
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=config.get("norm_obs", True),
            norm_reward=config.get("norm_reward", True),
            clip_obs=config.get("clip_obs", 3.0),
            clip_reward=config.get("clip_reward", 10.0),
        )

        # Setup policy network architecture
        policy_kwargs = {
            "net_arch": {
                "pi": config.get("policy_network", [256, 256, 128]),
                "vf": config.get("value_network", [256, 256, 128]),
            },
            "activation_fn": torch.nn.ReLU,
        }

        # Create model
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=config.get("learning_rate", 1e-3),
            n_steps=config.get("n_steps", 4096),
            batch_size=config.get("batch_size", 256),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            ent_coef=config.get("ent_coef", 0.05),
            vf_coef=config.get("vf_coef", 0.5),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            verbose=1,
            tensorboard_log=os.path.join(exp_dir, "tb_logs"),
        )

        # Setup callbacks
        metric_callback = MetricTrackingCallback(eval_env, eval_freq=config.get("eval_freq", 5000))
        entropy_callback = EntropySchedulerCallback(
            initial_ent=config.get("initial_ent", 0.05),
            final_ent=config.get("final_ent", 0.005),
            total_steps=total_timesteps,
        )
        tb_callback = TensorboardMetricsCallback(
            log_dir=exp_dir,
            starting_learning_rate=config.get("learning_rate", 1e-3),
            starting_clip_range=config.get("clip_range", 0.2),
        )

        # Train model
        print(f"Starting experiment: {experiment_name}")
        print(f"Configuration: {config}")
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            callback=[metric_callback, entropy_callback, tb_callback],
            progress_bar=True,
        )

        training_time = time.time() - start_time

        # Save model
        model_path = os.path.join(exp_dir, "final_model.zip")
        model.save(model_path)

        # Save metrics
        metrics_path = os.path.join(exp_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metric_callback.metrics, f, indent=2, cls=NumpyEncoder)

        # Plot learning curves
        self._plot_learning_curves(metric_callback.metrics, exp_dir)

        # Store results
        results = {
            "experiment_name": experiment_name,
            "config": config,
            "metrics": metric_callback.metrics,
            "training_time": training_time,
            "final_success_rate": (
                metric_callback.metrics["success_rate"][-1] if metric_callback.metrics["success_rate"] else None
            ),
            "final_mean_reward": (
                metric_callback.metrics["mean_reward"][-1] if metric_callback.metrics["mean_reward"] else None
            ),
            "final_ball_to_hole": (
                metric_callback.metrics["mean_dist_ball_to_hole"][-1]
                if metric_callback.metrics["mean_dist_ball_to_hole"]
                else None
            ),
        }

        self.results[experiment_name] = results

        # Clean up
        env.close()
        eval_env.close()

        return results

    # pylint: disable=redefined-outer-name
    def run_grid_search(self, param_grid, base_config=None, total_timesteps=1_000_000):
        """Run a grid search over hyperparameter combinations.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            base_config: Base configuration to use for all experiments
            total_timesteps: Total timesteps to train for each experiment

        Returns:
            DataFrame containing results for all experiments
        """
        if base_config is None:
            base_config = {}

        # Generate all combinations of parameters
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # Function to recursively generate all combinations
        def generate_configs(index=0, current_config=None):
            if current_config is None:
                current_config = base_config.copy()

            if index >= len(param_names):
                return [current_config.copy()]

            configs = []
            param_name = param_names[index]
            for value in param_values[index]:
                config_copy = current_config.copy()
                config_copy[param_name] = value
                configs.extend(generate_configs(index + 1, config_copy))

            return configs

        all_configs = generate_configs()
        print(f"Running grid search with {len(all_configs)} configurations")

        # Run each configuration
        for i, config in enumerate(all_configs):
            config_str = "_".join(f"{k}_{v}" for k, v in config.items() if k in param_grid)
            experiment_name = f"grid_search_{i}_{config_str}"
            self.run_experiment(config, experiment_name, total_timesteps)

        # Compile results into DataFrame
        return self.get_results_dataframe()

    def get_results_dataframe(self):
        """Convert experiment results to a pandas DataFrame for analysis."""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for exp_name, result in self.results.items():
            row = {
                "experiment_name": exp_name,
                "training_time": result["training_time"],
                "final_success_rate": result["final_success_rate"],
                "final_mean_reward": result["final_mean_reward"],
                "final_ball_to_hole": result["final_ball_to_hole"],
            }
            # Add config parameters
            for k, v in result["config"].items():
                row[f"config_{k}"] = v

            rows.append(row)

        return pd.DataFrame(rows)

    def _plot_learning_curves(self, metrics, exp_dir):
        """Plot learning curves from collected metrics."""
        _, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot mean reward
        axes[0, 0].plot(metrics["timesteps"], metrics["mean_reward"])
        axes[0, 0].set_title("Mean Reward")
        axes[0, 0].set_xlabel("Timesteps")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)

        # Plot ball-to-hole distance
        axes[0, 1].plot(metrics["timesteps"], metrics["mean_dist_ball_to_hole"])
        axes[0, 1].set_title("Mean Ball-to-Hole Distance")
        axes[0, 1].set_xlabel("Timesteps")
        axes[0, 1].set_ylabel("Distance")
        axes[0, 1].grid(True)

        # Plot club-to-ball distance
        axes[1, 0].plot(metrics["timesteps"], metrics["mean_dist_club_to_ball"])
        axes[1, 0].set_title("Mean Club-to-Ball Distance")
        axes[1, 0].set_xlabel("Timesteps")
        axes[1, 0].set_ylabel("Distance")
        axes[1, 0].grid(True)

        # Plot success rate
        axes[1, 1].plot(metrics["timesteps"], metrics["success_rate"])
        axes[1, 1].set_title("Success Rate")
        axes[1, 1].set_xlabel("Timesteps")
        axes[1, 1].set_ylabel("Success Rate")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "learning_curves.png"))
        plt.close()


def compare_experiments(experiment_manager, metric="final_mean_reward", ascending=False):
    """Compare experiments based on a specific metric.

    Args:
        experiment_manager: ExperimentManager instance
        metric: Metric to sort by
        ascending: Whether to sort in ascending order

    Returns:
        Sorted DataFrame of experiment results
    """
    df = experiment_manager.get_results_dataframe()
    return df.sort_values(by=metric, ascending=ascending)


# Example usage
if __name__ == "__main__":
    # Define base configuration
    base_config = {
        "learning_rate": 1e-3,
        "n_steps": 4096,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "include_velocities": False,
        "norm_obs": True,
        "norm_reward": True,
        "clip_obs": 3.0,
        "clip_reward": 10.0,
        "policy_network": [256, 256, 128],
        "value_network": [256, 256, 128],
        "initial_ent": 0.05,
        "final_ent": 0.005,
        "eval_freq": 5000,
    }

    # Define parameter grid for search
    param_grid = {"learning_rate": [1e-4, 3e-4, 1e-3], "batch_size": [128, 256, 512], "gamma": [0.98, 0.99, 0.995]}

    # Create experiment manager
    manager = ExperimentManager()

    # Option 1: Run a single experiment
    # manager.run_experiment(base_config, "baseline_experiment", total_timesteps=1_000_000)

    # Option 2: Run a grid search
    # results_df = manager.run_grid_search(param_grid, base_config, total_timesteps=500_000)

    # Compare and analyze results
    # best_experiments = compare_experiments(manager)
    # print("Best experiments by final mean reward:")
    # print(best_experiments)
