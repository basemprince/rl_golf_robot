"""This script tests the FrankaIkGolfCourseEnv-v0 environment using the SAIClient."""

import time

from sai_rl import SAIClient

# === Connect and create env ===
sai = SAIClient("FrankaIkGolfCourseEnv-v0")
env = sai.make_env(render_mode="human")

# === Reset environment ===
obs, info = env.reset()  # Gymnasium API
print("Observation shape:", obs.shape)
print("Initial observation:", obs)
print("Initial info:", info)

# === Random test steps ===
for step in range(20):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)  # Gymnasium API

    print(f"\nStep {step}")
    print("Action:", action)
    print("Reward:", reward)
    print("Terminated:", terminated, "Truncated:", truncated)
    print("Info:", info)

    # Look for task-specific keys
    if "grasp_success" in info:
        print("Grasp success:", info["grasp_success"])
    if "is_holding_club" in info:
        print("Holding club:", info["is_holding_club"])
    if "club_dropped" in info:
        print("Club dropped:", info["club_dropped"])

    time.sleep(0.5)

    if terminated or truncated:
        print("Episode ended. Resetting environment...")
        obs, info = env.reset()

env.close()
