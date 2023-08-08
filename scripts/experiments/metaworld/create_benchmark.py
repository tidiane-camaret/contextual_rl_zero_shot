import metaworld
import random

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks

env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`

print('nb of tasks : ' , len(ml1.train_tasks))  # 50 settings within a task
'''
for task in ml1.train_tasks:
    env.set_task(task)  # Set task
    print(env.obj_init_pos) # initial position is modified each time
'''

task = ml1.train_tasks[0] # sample a task
print(task.env_name)
#print(task)
env.set_task(task)  # Set task
print(env.model.body_mass[0], env.model.body_inertia[0])
env.model.body_mass[0] = 0.1
env.model.body_inertia[0] = [0.0001, 0.0001, 0.0001]
obs = env.reset()
print(env.model.body_mass[0], env.model.body_inertia[0])

### RENDERING
"""
for _ in range(500):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
"""

### TRAINING AN AGENT
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

#env = make_vec_env(lambda: env, n_envs=1)
env.max_path_length = 100000

model = PPO('MlpPolicy',
             env,
             n_steps=500, 
             verbose=1, 
             tensorboard_log="./ppo_pick_place_tensorboard/")
model.learn(total_timesteps=10000, reset_num_timesteps=True)

### EVALUATING THE AGENT
env = ml1.test_classes['pick-place-v2']() # Create an environment with task `pick_place`
env.set_task(task)  # Set task

obs = env.reset()
for _ in range(500):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

