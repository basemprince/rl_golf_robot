# pylint: disable=duplicate-code
"""
Visualization tools for experiment results.

This module provides functions to visualize and compare results from
different hyperparameter experiments.
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Create visualization directory
VISUALIZATION_DIR = "./visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)


def load_experiment_data(experiments_dir="./experiments"):
    """Load data from all experiments in the experiments directory.

    Args:
        experiments_dir: Directory containing experiment folders

    Returns:
        DataFrame containing experiment data
    """
    experiment_folders = [f for f in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, f))]

    all_data = []

    for folder in experiment_folders:
        folder_path = os.path.join(experiments_dir, folder)

        # Load config
        config_path = os.path.join(folder_path, "config.json")
        if not os.path.exists(config_path):
            continue

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load metrics
        metrics_path = os.path.join(folder_path, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        # Extract final metrics
        final_metrics = {
            "experiment_name": folder,
            "final_timestep": metrics["timesteps"][-1] if metrics["timesteps"] else None,
            "final_mean_reward": metrics["mean_reward"][-1] if metrics["mean_reward"] else None,
            "final_std_reward": metrics["std_reward"][-1] if metrics["std_reward"] else None,
            "final_ball_to_hole": metrics["mean_dist_ball_to_hole"][-1] if metrics["mean_dist_ball_to_hole"] else None,
            "final_club_to_ball": metrics["mean_dist_club_to_ball"][-1] if metrics["mean_dist_club_to_ball"] else None,
            "final_success_rate": metrics["success_rate"][-1] if metrics["success_rate"] else None,
            "metrics": metrics,
        }

        # Add config parameters
        for k, v in config.items():
            final_metrics[f"config_{k}"] = v

        all_data.append(final_metrics)

    if not all_data:
        return pd.DataFrame()

    return pd.DataFrame(all_data)


def plot_learning_curves(df, output_dir=VISUALIZATION_DIR):
    """Plot learning curves for all experiments.

    Args:
        df: DataFrame containing experiment data
        output_dir: Directory to save plots
    """
    if df.empty:
        print("No experiment data to plot")
        return

    # Create figure for all experiments
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Learning Curves for All Experiments", fontsize=16)

    # Plot metrics for each experiment
    for _, row in df.iterrows():
        metrics = row["metrics"]
        label = row["experiment_name"]

        # Mean reward
        axes[0, 0].plot(metrics["timesteps"], metrics["mean_reward"], label=label)

        # Ball-to-hole distance
        axes[0, 1].plot(metrics["timesteps"], metrics["mean_dist_ball_to_hole"], label=label)

        # Club-to-ball distance
        axes[1, 0].plot(metrics["timesteps"], metrics["mean_dist_club_to_ball"], label=label)

        # Success rate
        axes[1, 1].plot(metrics["timesteps"], metrics["success_rate"], label=label)

    # Set titles and labels
    axes[0, 0].set_title("Mean Reward")
    axes[0, 0].set_xlabel("Timesteps")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)

    axes[0, 1].set_title("Mean Ball-to-Hole Distance")
    axes[0, 1].set_xlabel("Timesteps")
    axes[0, 1].set_ylabel("Distance")
    axes[0, 1].grid(True)

    axes[1, 0].set_title("Mean Club-to-Ball Distance")
    axes[1, 0].set_xlabel("Timesteps")
    axes[1, 0].set_ylabel("Distance")
    axes[1, 0].grid(True)

    axes[1, 1].set_title("Success Rate")
    axes[1, 1].set_xlabel("Timesteps")
    axes[1, 1].set_ylabel("Success Rate")
    axes[1, 1].grid(True)

    # Add legend to a separate figure to avoid overcrowding
    fig.tight_layout()
    plt.figlegend(loc="lower center", ncol=3, labelspacing=0.0)
    plt.subplots_adjust(bottom=0.2)

    # Save figure
    plt.savefig(os.path.join(output_dir, "all_learning_curves.png"))
    plt.close()


def plot_parameter_comparison(df, param_name, metric="final_mean_reward", output_dir=VISUALIZATION_DIR):
    """Plot comparison of a parameter across experiments.

    Args:
        df: DataFrame containing experiment data
        param_name: Parameter to compare (without 'config_' prefix)
        metric: Metric to use for comparison
        output_dir: Directory to save plots
    """
    if df.empty:
        print("No experiment data to plot")
        return

    # Check if parameter exists in dataframe
    param_col = f"config_{param_name}"
    if param_col not in df.columns:
        print(f"Parameter {param_name} not found in experiment data")
        return

    # Group by parameter value and calculate mean and std of metric
    grouped = df.groupby(param_col)[metric].agg(["mean", "std"]).reset_index()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(grouped[param_col].astype(str), grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=5)
    plt.xlabel(param_name)
    plt.ylabel(metric)
    plt.title(f"Effect of {param_name} on {metric}")
    plt.grid(True)

    # Save figure
    plt.savefig(os.path.join(output_dir, f"{param_name}_comparison.png"))
    plt.close()


def create_correlation_heatmap(df, output_dir=VISUALIZATION_DIR):
    """Create a heatmap showing correlations between parameters and metrics.

    Args:
        df: DataFrame containing experiment data
        output_dir: Directory to save plots
    """
    if df.empty:
        print("No experiment data to plot")
        return

    # Extract numeric columns that are either metrics or config parameters
    numeric_cols = []
    for col in df.columns:
        if col.startswith("final_") or col.startswith("config_"):
            try:
                # Check if column can be converted to numeric
                pd.to_numeric(df[col])
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue

    if not numeric_cols:
        print("No numeric columns found for correlation analysis")
        return

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation between Parameters and Metrics")
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()


def plot_top_experiments(df, metric="final_mean_reward", n=5, output_dir=VISUALIZATION_DIR):
    """Plot the top N experiments based on a metric.

    Args:
        df: DataFrame containing experiment data
        metric: Metric to use for ranking
        n: Number of top experiments to show
        output_dir: Directory to save plots
    """
    if df.empty:
        print("No experiment data to plot")
        return None

    # Sort by metric and get top N
    top_df = df.sort_values(by=metric, ascending=False).head(n)

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_df["experiment_name"], top_df[metric])

    # Add values on top of bars
    for bar_element in bars:
        height = bar_element.get_height()
        plt.text(
            bar_element.get_x() + bar_element.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Experiment")
    plt.ylabel(metric)
    plt.title(f"Top {n} Experiments by {metric}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join(output_dir, f"top_{n}_experiments.png"))
    plt.close()

    # Also create a table with key parameters
    param_cols = [col for col in top_df.columns if col.startswith("config_")]
    metric_cols = [col for col in top_df.columns if col.startswith("final_")]

    selected_cols = ["experiment_name"] + param_cols + metric_cols
    top_params = top_df[selected_cols]

    # Save to CSV
    top_params.to_csv(os.path.join(output_dir, f"top_{n}_experiments.csv"), index=False)

    return top_params


def generate_comprehensive_report(output_dir=VISUALIZATION_DIR):
    """Generate a comprehensive report of all experiments.

    Args:
        output_dir: Directory to save the report
    """
    # Load experiment data
    df = load_experiment_data()

    if df.empty:
        print("No experiment data found")
        return

    # Create visualizations
    plot_learning_curves(df, output_dir)
    create_correlation_heatmap(df, output_dir)
    plot_top_experiments(df, metric="final_mean_reward", n=5, output_dir=output_dir)
    plot_top_experiments(df, metric="final_success_rate", n=5, output_dir=output_dir)

    # Find parameters that were varied across experiments
    varied_params = []
    for col in df.columns:
        if col.startswith("config_") and df[col].nunique() > 1:
            param_name = col[7:]  # Remove 'config_' prefix
            varied_params.append(param_name)
            plot_parameter_comparison(df, param_name, "final_mean_reward", output_dir)

    # Generate summary report
    report = [
        "# Experiment Results Summary",
        f"Total experiments: {len(df)}",
        "",
        "## Top Experiments by Mean Reward",
        df.sort_values("final_mean_reward", ascending=False)[["experiment_name", "final_mean_reward"]]
        .head(5)
        .to_string(),
        "",
        "## Top Experiments by Success Rate",
        df.sort_values("final_success_rate", ascending=False)[["experiment_name", "final_success_rate"]]
        .head(5)
        .to_string(),
        "",
        "## Parameters Varied",
        ", ".join(varied_params),
        "",
        "## Generated Visualizations",
        "- Learning curves for all experiments",
        "- Correlation heatmap between parameters and metrics",
        "- Top experiments by mean reward",
        "- Top experiments by success rate",
    ]

    for param in varied_params:
        report.append(f"- Effect of {param} on final mean reward")

    # Write report to file
    with open(os.path.join(output_dir, "experiment_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Comprehensive report generated in {output_dir}")


if __name__ == "__main__":
    generate_comprehensive_report()
