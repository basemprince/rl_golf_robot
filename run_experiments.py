"""
Script to run experiments with different hyperparameter configurations.

This script provides examples of how to use the ExperimentManager to run
single experiments or grid searches with different hyperparameter configurations.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from hyperparameter_tuning import ExperimentManager

# Create experiments directory if it doesn't exist
os.makedirs("./experiment_results", exist_ok=True)


def run_single_experiment():
    """Run a single experiment with a specific configuration."""
    manager = ExperimentManager()

    # Define configuration
    config = {
        "learning_rate": 3e-4,
        "n_steps": 4096,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "include_velocities": True,  # Try with velocities
        "norm_obs": True,
        "norm_reward": False,  # Match main.py
        "clip_obs": 5.0,  # Match main.py
        "clip_reward": 10.0,
        "policy_network": [256, 256, 128],
        "value_network": [256, 256, 128],
        "initial_ent": 0.05,
        "final_ent": 0.005,
        "eval_freq": 5000,
    }

    # Run experiment
    result = manager.run_experiment(config, experiment_name="velocity_experiment", total_timesteps=1_000_000)

    print(f"Experiment completed with final success rate: {result['final_success_rate']}")
    return result


def run_learning_rate_search():
    """Run a grid search over learning rates."""
    manager = ExperimentManager()

    # Base configuration
    base_config = {
        "n_steps": 4096,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "include_velocities": True,
        "norm_obs": True,
        "norm_reward": False,  # Match main.py
        "clip_obs": 5.0,  # Match main.py
        "clip_reward": 10.0,
        "policy_network": [256, 256, 128],
        "value_network": [256, 256, 128],
        "initial_ent": 0.05,
        "final_ent": 0.005,
        "eval_freq": 5000,
    }

    # Parameter grid
    param_grid = {"learning_rate": [1e-4, 3e-4, 1e-3, 3e-3]}

    # Run grid search with parallel execution
    results_df = manager.run_grid_search(
        param_grid, base_config, total_timesteps=500_000, n_jobs=4  # Use 4 parallel jobs
    )

    # Save results
    results_df.to_csv("./experiment_results/learning_rate_search.csv", index=False)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(results_df["config_learning_rate"].astype(str), results_df["final_mean_reward"])
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Mean Reward")
    plt.title("Learning Rate vs. Performance")
    plt.savefig("./experiment_results/learning_rate_comparison.png")

    return results_df


def run_network_architecture_search():
    """Run a grid search over network architectures."""
    manager = ExperimentManager()

    # Base configuration
    base_config = {
        "learning_rate": 3e-4,  # Use best learning rate from previous search
        "n_steps": 4096,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "include_velocities": True,
        "norm_obs": True,
        "norm_reward": False,  # Match main.py
        "clip_obs": 5.0,  # Match main.py
        "clip_reward": 10.0,
        "initial_ent": 0.05,
        "final_ent": 0.005,
        "eval_freq": 5000,
    }

    # Parameter grid for network architectures
    param_grid = {
        "policy_network": [[128, 128], [256, 256], [512, 256, 128], [256, 256, 256]],
        "value_network": [[128, 128], [256, 256], [512, 256, 128], [256, 256, 256]],
    }

    # Run grid search with parallel execution
    results_df = manager.run_grid_search(param_grid, base_config, total_timesteps=300_000, n_jobs=4)

    # Save results
    results_df.to_csv("./experiment_results/network_architecture_search.csv", index=False)

    return results_df


def run_comprehensive_search():
    """Run a comprehensive grid search over multiple parameters."""
    manager = ExperimentManager()

    # Base configuration
    base_config = {
        "n_steps": 4096,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "include_velocities": True,
        "norm_obs": True,
        "norm_reward": False,  # Match main.py
        "clip_obs": 5.0,  # Match main.py
        "clip_reward": 10.0,
        "policy_network": [256, 256, 128],
        "value_network": [256, 256, 128],
        "eval_freq": 5000,
    }

    # Parameter grid
    param_grid = {
        "learning_rate": [1e-4, 1e-5],
        "batch_size": [64, 128, 256, 512],
        "ent_coef": [0.02, 0.1],
        "n_steps": [128, 512],
        # "initial_ent": [0.05, 0.1],
    }

    # Run grid search with shorter training time and parallel execution
    results_df = manager.run_grid_search(
        param_grid, base_config, total_timesteps=300_000, n_jobs=4  # Use 4 parallel jobs
    )

    # Save results
    results_df.to_csv("./experiment_results/comprehensive_search.csv", index=False)

    return results_df


def analyze_results():
    """Analyze results from previous experiments."""
    # Load results from CSV files
    try:
        lr_results = pd.read_csv("./experiment_results/learning_rate_search.csv")
        print("Learning Rate Search Results:")
        print(lr_results[["experiment_name", "config_learning_rate", "final_mean_reward", "final_success_rate"]])
        print("\n")
    except FileNotFoundError:
        print("Learning rate search results not found.")

    try:
        net_results = pd.read_csv("./experiment_results/network_architecture_search.csv")
        print("Network Architecture Search Results:")
        print(net_results[["experiment_name", "config_policy_network", "final_mean_reward", "final_success_rate"]])
        print("\n")
    except FileNotFoundError:
        print("Network architecture search results not found.")

    try:
        comp_results = pd.read_csv("./experiment_results/comprehensive_search.csv")
        print("Comprehensive Search Results (Top 3):")
        top_results = comp_results.sort_values("final_mean_reward", ascending=False).head(3)
        print(top_results[["experiment_name", "final_mean_reward", "final_success_rate"]])

        # Print configuration of best experiment
        best_exp = top_results.iloc[0]
        print("\nBest Configuration:")
        for col in best_exp.index:
            if col.startswith("config_"):
                print(f"{col[7:]}: {best_exp[col]}")
    except FileNotFoundError:
        print("Comprehensive search results not found.")


if __name__ == "__main__":
    # Choose which experiment to run
    print("Select an experiment to run:")
    print("1. Single experiment")
    print("2. Learning rate search")
    print("3. Network architecture search")
    print("4. Comprehensive parameter search")
    print("5. Analyze previous results")

    choice = input("Enter your choice (1-5): ")

    if choice == "1":
        run_single_experiment()
    elif choice == "2":
        run_learning_rate_search()
    elif choice == "3":
        run_network_architecture_search()
    elif choice == "4":
        run_comprehensive_search()
    elif choice == "5":
        analyze_results()
    else:
        print("Invalid choice. Please run again with a valid option.")
