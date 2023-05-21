import gym
from stable_baselines3 import PPO
from meta_rl.interaction_policy.interaction_policy import EPI_PPO

env = gym.make("CartPole-v1", )

model = EPI_PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")