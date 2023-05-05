import argparse
import numpy as np

import meta_rl
import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor
from stable_baselines3 import A2C, PPO, DQN

import wandb
from wandb.integration.sb3 import WandbCallback

# main function with "oracle" as boolean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--nb_steps', type=int, default=2_000_000)
    parser.add_argument('--nb_evals', type=int, default=1000)
    args = parser.parse_args()
    oracle = args.oracle
    render = args.render
    nb_total_timesteps = args.nb_steps
    nb_evals = args.nb_evals

    print("Oracle: ", oracle)

    task_name = "striker"
    NUM_OF_PARAMS = 2

    run = wandb.init(
        project="meta_rl_epi_orig",
        monitor_gym=True, # auto-upload the videos of agents playing the game
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        config={
            "task_name": task_name,
            "oracle": oracle,
            "num_of_params": NUM_OF_PARAMS,
            "total_timesteps": nb_total_timesteps,
        }
        )

        

    # generate the training environment

    if oracle:
        train_env = gym.make("StrikerAvg-v0")
    else:
        train_env = gym.make("StrikerOracle-v0")

    model = PPO('MlpPolicy', 
                env=train_env,
                verbose=1,
                tensorboard_log="results/tensorboard/"+task_name+"/")


    model.learn(total_timesteps=nb_total_timesteps,
                callback=WandbCallback(),
                )
    """
    model.save("results/policies/" + task_name)

    model = PPO.load("results/policies/" + "striker_working" )# + task_name)
    """
    # evaluate the policy on an unseen scale value
    """

    eval_coeff_list = np.linspace(0.1, 2, 10)
    mean_reward_list = []
    std_reward_list = []

    for eval_coeff in eval_coeff_list:
        eval_env = gym.make('StrikerCustom-v0', eval_mode=True, oracle=oracle, eval_scale=[eval_coeff, eval_coeff])

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=nb_evals)
        mean_reward_list.append(mean_reward)
        std_reward_list.append(std_reward)
        print(f"eval_coeff: {eval_coeff}, mean_reward:{mean_reward:.2f} +/- {std_reward}")
    
    
    mean_reward_list = np.array(mean_reward_list)
    std_reward_list = np.array(std_reward_list)
    # log the results
    data = [[param, mean_reward] for param, mean_reward in zip(eval_coeff_list, mean_reward_list)]
    table = wandb.Table(data=data, columns=["param", "mean_reward"])
    wandb.log({"mean_reward_plot": wandb.plot.line(table, "param", "mean_reward")})

    # close wandb
    """
    run.finish()

    # render the policy

    eval_env = env

    obs = eval_env.reset()
    print("obs:", obs.shape, )

    if render:
        for _ in range(10):
            obs = eval_env.reset()
            for _ in range(100):
                action, _states = model.predict(obs)
                obs, reward, done, info = eval_env.step(action)
                eval_env.render()

