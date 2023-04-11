import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3 import A2C, DQN, SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor

import wandb
from wandb.integration.sb3 import WandbCallback

run = wandb.init(
    project="meta_rl",
    monitor_gym=True, # auto-upload the videos of agents playing the game
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

train_param_range = np.linspace(0.5, 2.5, 20)

train_env = vec_env.DummyVecEnv([
    lambda: monitor.Monitor(
    RecordEpisodeStatistics(CartPoleEnv(length=l, oracle=False)),
    )
 for l in train_param_range])  

model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="results/tensorboard/cartpole/")
model.learn(total_timesteps=100_000,
            callback=WandbCallback(),
            )
