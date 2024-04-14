import json
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_eval_violins(
    results_folder_path: str,
    context_modes="all",
    context_colors=None,
    environment_name="",
):
    """
    given a run directory, plots the rewards for each context value for each context mode
    """
    # run_dir = parent of current dir + run_dir
    run_dir = results_folder_path
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

    plt.figure(figsize=(20, 6))
    # print statistics for each column

    sns.violinplot(
        data=reward_df,
        x="context_value",
        y="reward",
        hue="context_mode",
        palette=context_colors,
        scale="width",
    )  # inner="quartile")


    if environment_name != "":
        plt.title(f"Rewards for each context value in {environment_name}")
    plt.savefig(f"results/plots/{environment_name}_violin.png", bbox_inches="tight")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the evaluation stats")
    parser.add_argument("--results_folder_path", type=str, default="results/hydra/multirun/2024-04-12/11-31-34", help="Path to the results folder")
    parser.add_argument("--environment_name", type=str, default="CARLCartpoleContinuous", help="Name of the environment")
    args = parser.parse_args()
    plot_eval_violins(args.results_folder_path)