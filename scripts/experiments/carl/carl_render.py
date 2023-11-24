import importlib

env_module = importlib.import_module("carl.envs")
env_id = "CARLMountainCar"
CARLEnv = getattr(env_module, env_id)


CARLEnv.render_mode = "human"
env = CARLEnv(
    # You can play with different gravity values here
    contexts={0: CARLEnv.get_default_context()},
    # obs_context_as_dict=False,
    # hide_context = True,
)

# run the experiment
s, info = env.reset()
print("full obs :", s)
print("observation shape:", s["obs"].shape)


env.render()


steps = 0
while steps < 100:
    a = env.action_space.sample()
    s, r, done, truncated, info = env.step(a)
    env.render()
    steps += 1
    if done:
        break

env.close()
"""
# train using stable-baselines3 dqn

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment

#env = make_vec_env(lambda: env, n_envs=1)

# Train the agent
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
"""
