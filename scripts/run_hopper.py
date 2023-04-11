import gym

"""
env = gym.make("Hopper-v2")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

vec_env = model.get_env()
vec_env.original_mass = 10.0
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
"""
import numpy as np
import matplotlib.pyplot as plt

import gym

from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TRPO

import wandb
from wandb.integration.sb3 import WandbCallback


run = wandb.init(
    project="meta_rl",
    monitor_gym=True, # auto-upload the videos of agents playing the game
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

run.config["task"] = "hopper"

#env = gym.make("CartPole-v1") # original environment

env = gym.make("Striker-v2")
env = gym.wrappers.RecordEpisodeStatistics(env)

original_param = env.model.body_mass

model = TRPO("MlpPolicy", env, verbose=1, tensorboard_log="results/tensorboard/hopper/")
model.learn(total_timesteps=200_000,
            callback=WandbCallback(
                #gradient_save_freq=100,
                #verbose=2,
                                    ),
            )

# evaluate the model on differet param values
param_range = np.linspace(0.1, 2, 20)
# reverse the order of the params
#param_range = param_range[::-1]

mean_reward_list = []
std_reward_list = []
for param in param_range:
    env_eval = gym.make("Striker-v2")
    env_eval.model.body_mass[:] = param * original_param
    mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=20)
    mean_reward_list.append(mean_reward)
    std_reward_list.append(std_reward)
    print(f"param: {param}, mean_reward:{mean_reward:.2f} +/- {std_reward}")
    del env_eval

mean_reward_list = np.array(mean_reward_list)
std_reward_list = np.array(std_reward_list)
# log the results
data = [[param, mean_reward] for param, mean_reward in zip(param_range, mean_reward_list)]
table = wandb.Table(data=data, columns=["param", "mean_reward"])
wandb.log({"mean_reward_plot": wandb.plot.line(table, "param", "mean_reward")})

# plot the results

plt.plot(param_range, mean_reward_list)
plt.fill_between(param_range, mean_reward_list - std_reward_list, mean_reward_list + std_reward_list, alpha=0.2)
plt.xlabel("param")
plt.ylabel("mean reward")
plt.title("CartPole-v1")
plt.savefig("results/plots/cartpole.png")
plt.show()

