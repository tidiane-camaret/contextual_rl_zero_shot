import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3 import A2C, DQN, SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor
from sb3_contrib import TRPO

import wandb
from wandb.integration.sb3 import WandbCallback

from meta_rl.striker import StrikerEnv

run = wandb.init(
    project="meta_rl",
    monitor_gym=True, # auto-upload the videos of agents playing the game
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

# generate the training environment

train_param_range = np.linspace(0.5, 2.5, 20)

train_env = vec_env.DummyVecEnv([
    lambda: monitor.Monitor(
    RecordEpisodeStatistics(StrikerEnv()),
    )
 for l in train_param_range])  


# learn the policy

model = TRPO("MlpPolicy", train_env, verbose=1, tensorboard_log="results/tensorboard/cartpole/")
model.learn(total_timesteps=1_000_000,
            callback=WandbCallback(),
            )

# render the policy

test_env = StrikerEnv()
obs = test_env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    test_env.render()
    if done:
      obs = test_env.reset()
