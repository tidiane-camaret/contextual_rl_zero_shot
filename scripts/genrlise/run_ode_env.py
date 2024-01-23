import torch
import jax
print(torch.cuda.is_available())
print(jax.devices())

from meta_rl.envs.genrlise.complex_ode_bounded_reward import ComplexODEBoundedReward

env = ComplexODEBoundedReward([1, 1], 1)
print("env.action_space : ", env.action_space)
print("env.observation_space : ", env.observation_space)
# print the complete type of the observation space
print("env.observation_space : ", type(env.observation_space))
# sample a random action
action = env.action_space.sample()
# step the environment with the sampled action
state, reward, done, info = env.step(action)
# print the state, reward, done and info
print("State: ", state)
print("Reward: ", reward)
print("Done: ", done)