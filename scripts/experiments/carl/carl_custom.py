import gymnasium as gym
import meta_rl # necessary to access the custom envs
from meta_rl.envs.carl_cartpole import CARLCartPoleContinuous

#env = gym.make('CartPoleContinuous-v0')
env = CARLCartPoleContinuous()

obs, info = env.reset()

for step in range(2):
    action = env.action_space.sample()
    print('action=', action)
    obs, reward, done, trunc, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done, 'info=', info)
    env.render()