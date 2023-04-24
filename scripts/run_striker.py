import numpy as np

import meta_rl
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
NUM_OF_ENVS = 5
TOTAL_TIMESTEPS = 1_000_000
NUM_EVALS = 100
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

# generate the training environment

scale_list = np.random.randint(0, 5, (NUM_OF_ENVS, NUM_OF_PARAMS, ))*0.1
# same shape, but filled with ones
#scale_list = np.ones((NUM_OF_ENVS, NUM_OF_PARAMS, ))*0.5

"""
train_env = gym.make('Striker-v2')

train_env = gym.make('StrikerCustom-v0', scale=scale_list[0], oracle=oracle)# learn the policy
"""

train_env = vec_env.DummyVecEnv([
    lambda: monitor.Monitor(
    RecordEpisodeStatistics(gym.make('StrikerCustom-v0', scale=scale, oracle=oracle)),
    )
    for scale in scale_list])


model = PPO('MlpPolicy', 
            env=train_env,
            verbose=1,
            tensorboard_log="results/tensorboard/"+task_name+"/")


model.learn(total_timesteps=TOTAL_TIMESTEPS,
            callback=WandbCallback(),
            )

model.save("results/policies/" + task_name)
"""
model = PPO.load("results/policies/" + "striker_working" )# + task_name)
"""
# evaluate the policy on an unseen scale value

eval_env = gym.make('StrikerCustom-v0', scale = [0.8,0.8], oracle=oracle)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=NUM_EVALS)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")
wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})

# close wandb
run.finish()

# render the policy

obs = eval_env.reset()
print("obs:", obs.shape, )

for _ in range(10):
    obs = eval_env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()

