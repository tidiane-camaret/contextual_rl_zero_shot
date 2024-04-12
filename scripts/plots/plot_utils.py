import json
import os
import importlib
import pandas as pd
import numpy as np
from rliable import plot_utils
def extract_rewards_from_run_dir(
    run_dir: str,
    context_modes="all",
    train_values=[],
    eval_values=[],
):
    """
    given a run directory, plots the rewards for each context value for each context mode
    """
    # run_dir = parent of current dir + run_dir
    run_dir = os.path.join(os.getcwd(), run_dir)

    reward_df = pd.DataFrame(
        columns=["context_mode", "seed", "context_value", "reward"]
    )

    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if file.startswith("sac_rewards_") and file.endswith(".json"):
                # files are named "sac_rewards_{env_id}_{context_mode}_{seed}.json"
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                    # data is a dict with {"context_value":rewards}
                    context_mode = file.split("_")[3] + (
                        "_" + file.split("_")[4]
                        if file.split("_")[3] == "learned"
                        else ""
                    )  # 'contextual' or 'noncontextual
                    seed = int(file.split("_")[-1].split(".")[0])
                    if context_modes == "all" or context_mode in context_modes:
                        for context_value, rewards in data.items():
                            for reward in rewards:
                                reward_df = reward_df._append(
                                    {
                                        "context_mode": context_mode,
                                        "seed": seed,
                                        "context_value": context_value,
                                        "reward": reward,
                                    },
                                    ignore_index=True,
                                )

    return reward_df

def eval_random_agent(env_name, context_name, context_value, max_steps, num_episodes=50):
    """
    Evaluate a random agent in the environment
    """
# env setup
    print("env_name : ", env_name)
    if env_name == "CARLCartPoleContinuous":
        # custom environment, we need to import it directly
        from meta_rl.envs.carl_cartpole import CARLCartPoleContinuous
        CARLEnv = CARLCartPoleContinuous
    else:
        env_module = importlib.import_module("carl.envs")
        CARLEnv = getattr(env_module, env_name)

    eval_context = CARLEnv.get_default_context()
    eval_context[context_name] = context_value
    env = CARLEnv(
        # You can play with different gravity values here
        contexts={0: eval_context},
    )

    rewards = []
    for _ in range(num_episodes):
        done = False
        episode_reward = 0
        state = env.reset()
        steps = 0
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            if steps > max_steps:
                break
        rewards.append(episode_reward)
    return np.mean(rewards)

if __name__ == "__main__":
    eval_random_agent("CARLCartPoleContinuous", "gravity", 70)