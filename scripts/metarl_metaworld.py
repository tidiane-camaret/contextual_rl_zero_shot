import argparse
import numpy as np
import itertools

import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor
from stable_baselines3 import PPO

import wandb
from wandb.integration.sb3 import WandbCallback

import meta_rl
from meta_rl.definitions import RESULTS_DIR
import metaworld

NUM_OF_PARAMS = 2
NUM_OF_ENVS = 8
task_name = 'pick-place-v2'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--context', type=str, default='explicit')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--nb_steps', type=int, default=2_000_000)
    parser.add_argument('--nb_runs_per_eval', type=int, default=100)
    args = parser.parse_args()
    context = args.context
    render = args.render
    nb_total_timesteps = args.nb_steps
    nb_runs_per_eval = args.nb_runs_per_eval

    ml1 = metaworld.ML1(task_name)
    """
    run = wandb.init(
    project="meta_rl_context",
    monitor_gym=True, # auto-upload the videos of agents playing the game
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    config={
        "task_name": task_name,
        "conext": context,
        "num_of_params": NUM_OF_PARAMS,
        "total_timesteps": nb_total_timesteps,
    },
    #save_dir = RESULTS_DIR,
        )
    
    env_list = [ml1.train_classes['pick-place-v2']().set_task(task) for task in ml1.train_tasks[:10]]

    train_env = vec_env.DummyVecEnv([
        lambda: monitor.Monitor(
        RecordEpisodeStatistics(env
            ),
        )
        for env in env_list])
    """
    train_env = ml1.train_classes[task_name]()
    train_env = train_env.set_task(ml1.train_tasks[0])
    
    model = PPO('MlpPolicy', 
            env=train_env,
            verbose=1,
            tensorboard_log=RESULTS_DIR / "tensorboard" / task_name )
    
    model.learn(total_timesteps=nb_total_timesteps, 
                callback=WandbCallback())
    
    # generate the evaluation environment
    eval_env = vec_env.DummyVecEnv([
        lambda: monitor.Monitor(
        RecordEpisodeStatistics(train_env.set_task(task)
            ),
        )
        for task in ml1.test_tasks[:10]])
    
    # evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=nb_runs_per_eval,
        render=render,
        deterministic=True,
        return_episode_rewards=True,

    )