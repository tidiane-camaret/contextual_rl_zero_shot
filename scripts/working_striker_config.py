import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3 import PPO
from stable_baselines3.common import vec_env, monitor

import wandb
from wandb.integration.sb3 import WandbCallback

task_name = "striker"
"""
run = wandb.init(
    project="meta_rl",
    monitor_gym=True, # auto-upload the videos of agents playing the game
    sync_tensorboard=True, )
"""

env = monitor.Monitor(
    RecordEpisodeStatistics(
    gym.make('Striker-v2')
    )
    )

model = PPO('MlpPolicy', 
            env, 
            verbose=1,
            tensorboard_log="results/tensorboard/"+task_name+"/")

model.learn(total_timesteps=400_000,
             callback=WandbCallback()) 


for j in range(20):
    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()