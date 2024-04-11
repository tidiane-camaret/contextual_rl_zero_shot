import gymnasium as gym
from carl.envs import CARLBraxAnt

env = CARLBraxAnt()
run_name = "carl_brax_ant"
env = gym.wrappers.RecordVideo(env, f"results/videos/{run_name}")

env.reset()
for _ in range(100):
    env.step(env.action_space.sample())
env.close()