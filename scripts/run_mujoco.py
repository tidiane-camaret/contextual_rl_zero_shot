import numpy as np

import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import TRPO

import wandb
from wandb.integration.sb3 import WandbCallback

from meta_rl.striker_custom import CustomStrikerEnv as StrikerEnv


task_name = "striker"
NUM_OF_PARAMS = 2
NUM_OF_ENVS = 20
TOTAL_TIMESTEPS = 10_000
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

train_env = vec_env.DummyVecEnv([
    lambda: monitor.Monitor(
    RecordEpisodeStatistics(StrikerEnv(scale=scale, oracle=oracle)),
    )
 for scale in scale_list])  


# learn the policy
"""
hidden_sizes = (32, 32)
policy = MlpPolicy(
    name="policy",
    env_spec=train_env.spec,
    hidden_sizes=hidden_sizes,
)
"""
model = PPO(policy="MlpPolicy", 
             env = train_env,
             verbose=1, 
             batch_size = 100000,
             n_steps=200,
             gae_lambda = 1,
             tensorboard_log="results/tensorboard/"+task_name+"/")


model.learn(total_timesteps=TOTAL_TIMESTEPS,
            callback=WandbCallback(),
            )

model.save(task_name)

#model = TRPO.load("striker")

# evaluate the policy on an unseen scale value

eval_env = train_env

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")
wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})

# close wandb
run.finish()

# render the policy

obs = eval_env.reset()
print("obs:", obs.shape, )

for _ in range(500):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)
    eval_env.render()
    if done:
      obs = eval_env.reset()
