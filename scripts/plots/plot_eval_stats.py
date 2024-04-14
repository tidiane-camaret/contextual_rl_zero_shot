import os
import yaml
import pandas as pd
import numpy as np
import argparse
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from scripts.plots.plot_utils import extract_rewards_from_run_dir, eval_random_agent

def plot_eval_stats(results_folder_path: str):
    import matplotlib.pyplot as plt

    # each subfolder in the results folder corresponds to a different environment
    env_folders = [f.path for f in os.scandir(results_folder_path) if f.is_dir()]
    random_performance_df = pd.DataFrame(columns=["Environment", "context_value", "reward"])
    df_list = []

    for env_folder in env_folders:
        # access the config.yaml file to get the env config

        if os.path.exists(env_folder + "/default_value/0/0/.hydra/config.yaml"):
            with open(env_folder + "/default_value/0/0/.hydra/config.yaml") as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
                env_config = config["env"]

                env_name = env_config["id"]
                max_steps = env_config["max_episode_steps"]
                context_name = env_config["context"]["name"]
                eval_values = env_config["context"]["eval_values"]
                train_values = env_config["context"]["train_values"]                
                
                # calculate the random performance for the environment
                for context_value in eval_values:
                    random_performance = eval_random_agent(env_name, context_name, context_value, max_steps)
                    random_performance_df = random_performance_df._append({"Environment": env_name, "context_value": context_value, "reward": random_performance}, ignore_index=True)
                    print(f"Random performance for {env_name} with context value {context_value} is {random_performance}")
                

            # extract the rewards from the runs
            rewards_df = extract_rewards_from_run_dir(env_folder)
                
            # normalize the rewards
            # loop over context_mode and context_values
            for context_value in rewards_df["context_value"].unique():
                default_performance = rewards_df[(rewards_df["context_mode"] == "default") & (rewards_df["context_value"] == context_value)]["reward"]
                default_performance = default_performance.mean()
                for context_mode in rewards_df["context_mode"].unique():
                    # find the random performance for this context
                    # Create boolean masks for the conditions
                    condition_mask = (rewards_df["context_mode"] == context_mode) & (rewards_df["context_value"] == context_value)
                    random_performance = random_performance_df[(random_performance_df["Environment"] == env_name) & (random_performance_df["context_value"] == float(context_value))]["reward"].values[0]
                    # Apply the conditions and perform the calculation

                    normalized_reward = (rewards_df.loc[condition_mask, "reward"] - random_performance) / (default_performance - random_performance)
                    rewards_df.loc[condition_mask, "reward"] = normalized_reward
            rewards_df["Environment"] = env_name
            #print type of  rewards_df["context_value"]
            rewards_df["context_value"] = rewards_df["context_value"].astype(float)
            rewards_df["inter_extra"] = rewards_df["context_value"].apply(lambda x: "inter" if (x >= min(train_values)) and (x <= max(train_values)) else "extra")        
            print(rewards_df["inter_extra"].unique())
            df_list.append(rewards_df)

    rewards_df = pd.concat(df_list)


    # change the context_mode names : learned_jrpc -> jcpl, learned_iida -> predictive_id
    rewards_df["context_mode"] = rewards_df["context_mode"].replace({"learned_jcpl": "jcpl", "learned_iida": "predictive_id"})
    rewards_df = rewards_df[rewards_df["context_mode"].isin(["explicit", "hidden", "jcpl", "predictive_id"])]
    list_context_ranges = ['all', "inter", "extra"]

    # plot the results
    stats_df = pd.DataFrame(columns=["Environment", "context_mode","context_range", "metric", "value" ])
    for env in rewards_df["Environment"].unique():
        print(env)
        for context_range in list_context_ranges:
                print(context_range)
                filtered_reward_df = rewards_df
                filtered_reward_df = filtered_reward_df[filtered_reward_df["Environment"] == env]
                print(filtered_reward_df["inter_extra"].unique())
                filtered_reward_df = filtered_reward_df[filtered_reward_df["inter_extra"] == context_range] if context_range != "all" else filtered_reward_df
                print(filtered_reward_df)

                # put scores in a dict of shape {context_mode: np.array(n_seeds, n_envs)}
                algo_scores = {}
                for context_mode in filtered_reward_df["context_mode"].unique():
                    # get a numpy array of the scores for each algorithm, of shape (n_seeds, n_envs)
                    unique_context_values = list(filtered_reward_df[filtered_reward_df["context_mode"] == context_mode]["context_value"].unique())
                    scores = []
                    for context_value in unique_context_values:
                        scores_context_value = filtered_reward_df[(filtered_reward_df["context_mode"] == context_mode) & (filtered_reward_df["context_value"] == context_value)]["reward"].values
                        scores.append(scores_context_value)
                    scores = np.array(scores)
                    algo_scores[context_mode] = scores

                for context_mode, scores in algo_scores.items():
                    print(f"Context mode: {context_mode}")
                    print(f"score shape: {scores.shape}")
                    #print(scores)



                colors =  (125, 84, 178, 1), (218, 76, 76, 1),(71,154,95,1), (237, 183, 50, 1)#, (83, 135, 221, 1)
                # inverse order of the colors
                colors = colors[::-1]
                # need to divide by 255 to get the right colors
                colors = [(r/255, g/255, b/255, a) for r, g, b, a in colors]


                ## Aggregate metrics with 95% Stratified Bootstrap CIs

                # Load ALE scores as a dictionary mapping algorithms to their human normalized
                # score matrices, each of which is of size `(num_runs x num_games)`
                algorithms = list(algo_scores.keys())

                aggregate_func = lambda x: np.array([metrics.aggregate_iqm(x),])
                aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
                algo_scores, aggregate_func, reps=50000)
                fig, axes = plot_utils.plot_interval_estimates(
                aggregate_scores, aggregate_score_cis,
                metric_names=['IQM',],
                algorithms=algorithms, xlabel='',
                colors=dict(zip(algorithms, colors)))

                # Save the figure
                plt.savefig(f"results/plots/{env}_{context_range}_iqm.png", bbox_inches='tight')
                # close the figure
                plt.close()

                for context_mode in ["jcpl", "predictive_id", "explicit", "hidden"]:
                    stats_df = stats_df._append({"Environment": env, "context_mode": context_mode, "context_range": context_range, "metric": "IQM", "value": aggregate_scores[context_mode][0]}, ignore_index=True)

                #Probability of Improvement

                # Load ProcGen scores as a dictionary containing pairs of normalized score
                # matrices for pairs of algorithms we want to compare
                algorithm_pairs = {a1 + ', ' + a2: (algo_scores[a1], algo_scores[a2])
                                    for a1, a2 in [('jcpl', 'explicit'),
                                                    ('jcpl','hidden'),
                                                    ('jcpl', 'predictive_id'),
                                                    ]}
                average_probabilities, average_prob_cis = rly.get_interval_estimates(
                algorithm_pairs, metrics.probability_of_improvement, reps=200)
                plot_utils.plot_probability_of_improvement(average_probabilities, average_prob_cis, colors = [colors[2], colors[1], colors[0]])
                # Save the figure
                plt.savefig(f"results/plots/{env}_{context_range}_poi.png", bbox_inches='tight')
                # close the figure
                plt.close()

                for context_mode in ["jcpl, explicit", "jcpl, hidden", "jcpl, predictive_id"]:
                    stats_df = stats_df._append({"Environment": env, "context_mode": context_mode, "context_range": context_range, "metric": "POI", "value": average_probabilities[context_mode]}, ignore_index=True)


                ## Performance Profiles
                import matplotlib.pyplot as plt
                import seaborn as sns
                # Load ALE scores as a dictionary mapping algorithms to their human normalized
                # score matrices, each of which is of size `(num_runs x num_games)`.

                # Human normalized score thresholds
                thresholds = np.linspace(-1,2, 100)
                score_distributions, score_distributions_cis = rly.create_performance_profile(
                    algo_scores, thresholds)
                # Plot score distributions
                fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
                plot_utils.plot_performance_profiles(
                score_distributions, thresholds,
                performance_profile_cis=score_distributions_cis,
                colors=dict(zip(algorithms, colors)),
                xlabel=r'Normalized Score $(\tau)$',
                ax=ax)

                # Save the figure
                plt.savefig(f"results/plots/{env}_{context_range}_pp.png", bbox_inches='tight')
                # close the figure
                plt.close()

    # save the stats
    stats_df.to_csv("results/plots/stats.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the evaluation stats")
    parser.add_argument("--results_folder_path", type=str, default="results/hydra/multirun/2024-04-12/11-31-34", help="Path to the results folder")
    args = parser.parse_args()
    plot_eval_stats(args.results_folder_path)