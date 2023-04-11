import numpy as np
import matplotlib.pyplot as plt

import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3 import A2C, DQN, SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor

import wandb
from wandb.integration.sb3 import WandbCallback

from meta_rl.cartpole_custom import CartPoleEnv

run = wandb.init(
    project="meta_rl",
    monitor_gym=True, # auto-upload the videos of agents playing the game
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    )

#env = gym.make("CartPole-v1") # original environment


#env = CartPoleEnv(length=0.5, oracle=False)
#env = gym.wrappers.RecordEpisodeStatistics(env)

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


# evaluate the model on a different param value
env_eval = CartPoleEnv(length=4, oracle=False)
mean_reward, std_reward = evaluate_policy(model, env_eval, n_eval_episodes=1000)



# now, train the model in the oracle setting
del model
train_env = vec_env.DummyVecEnv([
    lambda: monitor.Monitor(
    RecordEpisodeStatistics(CartPoleEnv(length=l, oracle=True)),
    )
    for l in train_param_range])

model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="results/tensorboard/cartpole/")
model.learn(total_timesteps=100_000,
            callback=WandbCallback(),
            )

# evaluate the model on a different param value
env_eval = CartPoleEnv(length=4, oracle=True)
mean_reward_oracle, std_reward_oracle = evaluate_policy(model, env_eval, n_eval_episodes=1000)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")
print(f"mean_reward_oracle:{mean_reward_oracle:.2f} +/- {std_reward_oracle}")

run.log({"mean_reward": mean_reward, "std_reward": std_reward})
run.log({"mean_reward_oracle": mean_reward_oracle, "std_reward_oracle": std_reward_oracle})

"""
# evaluate the model on differet param values
param_range = np.linspace(0.5, 2.5, 4)
# reverse the order of the params
#param_range = param_range[::-1]

mean_reward_list = []
std_reward_list = []
for param in param_range:
    env_eval = CartPoleEnv(length=param)
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

"""




"""
# render the model

def render_model(env, model):
    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()#mode = "human")
        # VecEnv resets automatically
        if done:
            obs = env.reset()

#render_model(env = env,   model = model)
#render_model(env = CartPoleEnv(param=3),   model = model,)
"""
run.finish()