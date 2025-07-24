# RL Golf Robot

A reinforcement learning project for training a Franka robot arm to play golf using PPO (Proximal Policy Optimization).

## Project Structure

- `main.py`: Core training implementation with PPO
- `hyperparameter_tuning.py`: Framework for running experiments with different hyperparameters
- `run_experiments.py`: Script to run various experiment configurations
- `visualize_results.py`: Tools for analyzing and visualizing experiment results

## Hyperparameter Experimentation

This project includes a comprehensive framework for running experiments with different hyperparameter configurations and tracking their performance.

### Key Features

- Run individual experiments with custom configurations
- Perform grid searches over multiple hyperparameter combinations
- Track and compare metrics across experiments
- Visualize learning curves and performance metrics
- Generate comprehensive reports on experiment results

### Available Metrics

The framework tracks several key metrics:

- **Mean reward**: Overall performance measure
- **Success rate**: Percentage of episodes where the ball gets close to the hole
- **Ball-to-hole distance**: Average final distance from ball to hole
- **Club-to-ball distance**: Average final distance from club to ball

### Running Experiments

To run experiments, use the `run_experiments.py` script:

```bash
python run_experiments.py
```

This will present options to:
1. Run a single experiment with a specific configuration
2. Run a learning rate search
3. Run a network architecture search
4. Run a comprehensive parameter search
5. Analyze previous results

### Visualizing Results

To generate visualizations and reports from experiment results:

```bash
python visualize_results.py
```

This will create:
- Learning curves for all experiments
- Parameter comparison plots
- Correlation heatmaps
- Rankings of top-performing experiments
- A comprehensive summary report

## Hyperparameters

The following hyperparameters can be tuned:

### PPO Parameters
- `learning_rate`: Learning rate for the optimizer
- `n_steps`: Number of steps to run for each environment per update
- `batch_size`: Minibatch size for optimization
- `gamma`: Discount factor
- `gae_lambda`: Factor for trade-off of bias vs variance in GAE
- `clip_range`: Clipping parameter for PPO
- `ent_coef`: Entropy coefficient for exploration
- `vf_coef`: Value function coefficient
- `max_grad_norm`: Maximum norm for gradient clipping

### Network Architecture
- `policy_network`: Hidden layer sizes for policy network
- `value_network`: Hidden layer sizes for value function network

### Environment Parameters
- `include_velocities`: Whether to include joint velocities in observations
- `norm_obs`: Whether to normalize observations
- `norm_reward`: Whether to normalize rewards
- `clip_obs`: Maximum value for observation normalization
- `clip_reward`: Maximum value for reward normalization

### Entropy Scheduling
- `initial_ent`: Initial entropy coefficient
- `final_ent`: Final entropy coefficient

## Example Configuration

```python
config = {
    'learning_rate': 3e-4,
    'n_steps': 4096,
    'batch_size': 256,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.05,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'include_velocities': True,
    'norm_obs': True,
    'norm_reward': True,
    'clip_obs': 3.0,
    'clip_reward': 10.0,
    'policy_network': [256, 256, 128],
    'value_network': [256, 256, 128],
    'initial_ent': 0.05,
    'final_ent': 0.005,
    'eval_freq': 5000
}
```
