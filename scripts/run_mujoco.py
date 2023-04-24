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
TOTAL_TIMESTEPS = 1_000_000
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

#scale_list = np.random.randint(0, 5, (NUM_OF_ENVS, NUM_OF_PARAMS, ))*0.1
# same shape, but filled with ones
scale_list = np.ones((NUM_OF_ENVS, NUM_OF_PARAMS, ))*0.5
"""
train_env = vec_env.DummyVecEnv([
    lambda: monitor.Monitor(
    RecordEpisodeStatistics(StrikerEnv(scale=scale, oracle=oracle)),
    )
 for scale in scale_list])  
"""
train_env = vec_env.DummyVecEnv([
    lambda: monitor.Monitor(
    RecordEpisodeStatistics(gym.make('Striker-v2')),
    )
    for scale in scale_list])

train_env = gym.make('Striker-v2')

# learn the policy
"""
hidden_sizes = (32, 32)
policy = MlpPolicy(
    name="policy",
    env_spec=train_env.spec,
    hidden_sizes=hidden_sizes,
)
"""
model = PPO('MlpPolicy', 
            env=train_env,
            verbose=1,
            tensorboard_log="results/tensorboard/"+task_name+"/")

"""
model.learn(total_timesteps=TOTAL_TIMESTEPS,
            callback=WandbCallback(),
            )

model.save("results/policies/" + task_name)
"""
model = PPO.load("results/policies/" + "striker_working" )# + task_name)

# evaluate the policy on an unseen scale value

eval_env = gym.make('Striker-v2')  #StrikerEnv(scale = [0.5,0.5], oracle=oracle)
"""
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")
wandb.log({"mean_reward": mean_reward, "std_reward": std_reward})
"""
# close wandb
run.finish()

# render the policy

obs = eval_env.reset()
print("obs:", obs.shape, )

for _ in range(20):
    obs = eval_env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, info = eval_env.step(action)
        eval_env.render()

