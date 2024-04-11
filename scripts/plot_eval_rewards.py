from plot_utils import extract_rewards_from_run_dir, eval_random_agent
import pandas as pd
import os

print(os.getcwd())
result_dirs = {
        #"CARLCartPoleContinuous" : ["results/hydra/multirun/2024-03-08/09-16-26","tau", [0.02      , 0.02583099, 0.03336201, 0.04308869, 0.05565119, 0.07187627, 0.09283178, 0.11989685, 0.15485274, 0.2       ], 200]#[0.002     , 0.0033362 , 0.00556512, 0.00928318, 0.01548527, 0.02583099, 0.04308869, 0.07187627, 0.11989685, 0.2       ], 200],
        "CARLMountainCarContinuous"  : ["results/hydra/multirun/2024-03-08/14-15-04","power", [-0.015     , -0.01166667, -0.00833333, -0.005     , -0.00166667, 0.00166667,  0.005     ,  0.00833333,  0.01166667,  0.015     ] , 999],
}

random_performance_df = pd.read_csv("scripts/plots/random_performance.csv")

intra_extra = {
        #"CARLCartPoleContinuous" : {"intra":[ 0.04308869, 0.05565119, 0.07187627, 0.09283178], "extra":[0.02      , 0.02583099, 0.03336201 , 0.11989685, 0.15485274, 0.2       ]},
        #"CARLBraxAnt": {"intra":[30, 40, 50, 60, 70,], "extra":[1, 10, 20, 80, 90, 100]},
        "CARLMountainCarContinuous"  : {"intra":[  -0.005     , -0.00166667, 0.00166667,  0.005  ], "extra":[-0.015     , -0.01166667, -0.00833333,  0.00833333,  0.01166667,  0.015     ]},
        #"CARLPendulum" : {"intra":[ 0.1       ,  0.21544347,   0.46415888,  1.      ], "extra":[0.01      ,  0.02154435,  0.04641589, 2.15443469,  4.64158883, 10.]},
}


random_performance_df = pd.DataFrame(columns=["Environment", "context_value", "reward"])
for env_name, (dir, context_name, context_values, max_steps) in result_dirs.items():
    for context_value in context_values:
        random_performance = eval_random_agent(env_name, context_name, context_value, max_steps)
        random_performance_df = random_performance_df._append({"Environment": env_name, "context_value": context_value, "reward": random_performance}, ignore_index=True)
        print(f"Random performance for {env_name} with context value {context_value} is {random_performance}")
# concat all rewards, normalize using random and default agents
df_list = []
for env, (dir, _, _, _)  in result_dirs.items():
    print(f"Environment: {env}")
    
    reward_df = extract_rewards_from_run_dir(dir, environment_name=env)
    print(reward_df)
    # normalize the rewards
    # loop over context_mode and context_values
    for context_value in reward_df["context_value"].unique():
        default_performance = reward_df[(reward_df["context_mode"] == "default") & (reward_df["context_value"] == context_value)]["reward"]
        default_performance = default_performance.mean()
        for context_mode in reward_df["context_mode"].unique():
            # find the random performance for this context
            # Create boolean masks for the conditions
            condition_mask = (reward_df["context_mode"] == context_mode) & (reward_df["context_value"] == context_value)
            random_performance = random_performance_df[(random_performance_df["Environment"] == env) & (random_performance_df["context_value"] == float(context_value))]["reward"].values[0]
            # Apply the conditions and perform the calculation
            
            #print(reward_df.loc[condition_mask, "reward"])
            normalized_reward = (reward_df.loc[condition_mask, "reward"] - random_performance) / (default_performance - random_performance)
            reward_df.loc[condition_mask, "reward"] = normalized_reward
    reward_df["Environment"] = env
    df_list.append(reward_df)

reward_df = pd.concat(df_list)


# add intra_extra label
def get_intra_extra_label(row):
    env = row["Environment"]
    context_value = float(row["context_value"])

    if context_value in intra_extra[env]["intra"]:
        return "intra"
    elif context_value in intra_extra[env]["extra"]:
        return "extra"
    else:
        raise ValueError(f"Context value {context_value} is not in either intra or extra for environment {env}")
# Apply the function to each row along the axis
reward_df["intra_extra"] = reward_df.apply(get_intra_extra_label, axis=1)
        
# change the context_mode values : learned_jrpc -> jcpl, learned_iida -> predictive_id
reward_df["context_mode"] = reward_df["context_mode"].replace({"learned_jrpl": "jcpl", "learned_iida": "predictive_id"})
reward_df = reward_df[reward_df["context_mode"].isin(["explicit", "hidden", "jcpl", "predictive_id"])]


import numpy as np
import matplotlib.pyplot as plt
list_envs = result_dirs.keys()
list_context_ranges = ['all', "intra", "extra"]

stats_df = pd.DataFrame(columns=["Environment", "context_mode","context_range", "metric", "value" ])
# all pairs of envs and contexts
for env in list_envs:
    for context_range in list_context_ranges:
        filtered_reward_df = reward_df
        filtered_reward_df = filtered_reward_df[filtered_reward_df["Environment"] == env]
        filtered_reward_df = filtered_reward_df[filtered_reward_df["intra_extra"] == context_range] if context_range != "all" else filtered_reward_df


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

        from rliable import library as rly
        from rliable import metrics
        from rliable import plot_utils
        import numpy as np

        colors =  (125, 84, 178, 1), (218, 76, 76, 1),(71,154,95,1), (237, 183, 50, 1)#, (83, 135, 221, 1)
        # inverse order of the colors
        colors = colors[::-1]
        # need to divide by 255 to get the right colors
        colors = [(r/255, g/255, b/255, a) for r, g, b, a in colors]


        ## Aggregate metrics with 95% Stratified Bootstrap CIs

        # Load ALE scores as a dictionary mapping algorithms to their human normalized
        # score matrices, each of which is of size `(num_runs x num_games)`
        algorithms = list(algo_scores.keys())

        aggregate_func = lambda x: np.array([
        #metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        #metrics.aggregate_mean(x),
        #metrics.aggregate_optimality_gap(x)
        ])
        aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        algo_scores, aggregate_func, reps=50000)
        fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores, aggregate_score_cis,
        #metric_names=['Median', 'IQM', 'Mean', 'Optimality Gap'],
        metric_names=['IQM',],
        algorithms=algorithms, xlabel='',
        colors=dict(zip(algorithms, colors)))

        # Save the figure
        plt.savefig(f"results/eval_metrics_3/{env}_{context_range}_iqm.png", bbox_inches='tight')
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
        plt.savefig(f"results/eval_metrics_3/{env}_{context_range}_poi.png", bbox_inches='tight')
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
        plt.savefig(f"results/eval_metrics_3/{env}_{context_range}_pp.png", bbox_inches='tight')
        # close the figure
        plt.close()

# save the stats
stats_df.to_csv("results/eval_metrics_3/stats.csv", index=False)
