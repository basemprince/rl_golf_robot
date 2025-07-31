"""Module for training behavioral cloning models."""

import numpy as np
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
from sai_rl import SAIClient
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.ppo import MlpPolicy
from torch import nn

from main import FrankaFK, GolfRewardWrapper

# === Load dataset ===
data = np.load("expert_demo_3.npz")
observations = data["observations"]
actions = data["actions"]

# === Add noise to actions for robustness ===
noise_std = 0.05  # Standard deviation of Gaussian noise
print(f"Adding Gaussian noise (std={noise_std}) to {len(actions)} actions...")
noisy_actions = actions + np.random.normal(0, noise_std, actions.shape)
# Clip actions to reasonable bounds (assuming action space is roughly [-1, 1])
noisy_actions = np.clip(noisy_actions, -1.0, 1.0)
actions = noisy_actions

# === Create dummy environment for policy training ===
sai = SAIClient(comp_id="franka-ml-hiring")
raw_env = sai.make_env()
fk_solver = FrankaFK(raw_env.spec)
wrapped_env = GolfRewardWrapper(raw_env, include_velocities=True, fk_solver=fk_solver)

# === VecNormalize (important for PPO compatibility later) ===
venv = DummyVecEnv([lambda: wrapped_env])
venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=5.0)
venv.training = True  # enable stat updates

# === Wrap into Transitions ===
transitions = Transitions(
    obs=observations,
    acts=actions,
    next_obs=observations,  # fine for BC
    dones=np.zeros(len(actions), dtype=bool),
    infos=[{} for _ in range(len(actions))],
)

# === Policy instantiation ===
lr = 3e-5
lr_schedule = get_schedule_fn(lr)

# Use policy_kwargs to add dropout and regularization
policy_kwargs = {
    "net_arch": [512, 512, 256],
    "activation_fn": nn.ReLU,
    "ortho_init": False,  # Better for BC training
}

policy = MlpPolicy(
    observation_space=venv.observation_space, action_space=venv.action_space, lr_schedule=lr_schedule, **policy_kwargs
)

# === BC Trainer ===
bc_trainer = BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=transitions,
    policy=policy,
    rng=np.random.default_rng(42),
    optimizer_kwargs={"lr": lr},
    l2_weight=1e-4,
)

# === Train ===
epochs = 200  # Reduced from 400 to prevent overfitting
print(f"\n🚀 Starting BC training for {epochs} epochs...")
bc_trainer.train(n_epochs=epochs, log_interval=50, progress_bar=True)

# === Save artifacts ===
bc_trainer.policy.save("bc_policy.zip")
venv.save("vecnormalize.pkl")
print("\n✅ Training complete! Saved policy to bc_policy.zip and vecnormalize.pkl")
