"""Test Brax environments
"""

import os
from carl.envs import CARLBraxAnt

env = CARLBraxAnt()

s, info = env.reset()
print("obs_context :", s)
print("observation shape:", s["obs"].shape)
print("context:", s["context"])

for i in range(1):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print("obs_context :", obs)
    print("reward :", reward)
    print("done :", done)
    print("info :", info)
    env.render()