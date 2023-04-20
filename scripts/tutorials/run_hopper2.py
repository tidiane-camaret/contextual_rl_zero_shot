import gym
import numpy as np

from sb3_contrib import TRPO

env = gym.make("Striker-v2")
#env = gym.make("HopperOriginal-v0")
env = gym.wrappers.RecordEpisodeStatistics(env)
body_mass = env.model.body_mass

print("body_mass", body_mass)
# multiply the body mass by 2
#env.model.body_mass[:] = 2 * body_mass

model = TRPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000, log_interval=4)

#model.save("trpo_pendulum")
#del model # remove to demonstrate saving and loading
#model = TRPO.load("trpo_pendulum")

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()