import numpy as np

import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import TRPO

import wandb
from wandb.integration.sb3 import WandbCallback

from meta_rl.striker_custom import OriginalStrikerEnv as StrikerEnv

task_name = "striker"
NUM_OF_PARAMS = 2
NUM_OF_ENVS = 50
TOTAL_TIMESTEPS = 400_000
oracle = True

run = wandb.init(
    project="meta_rl",
    monitor_gym=True, # auto-upload the videos of agents playing the game
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    config={
        "task_name": task_name,
        "oracle": oracle,
        "num_of_params": NUM_OF_PARAMS,
        "num_of_envs": NUM_OF_ENVS,
        "total_timesteps": TOTAL_TIMESTEPS,
    }
    )