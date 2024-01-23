from meta_rl.envs.genrlise.complex_ode_bounded_reward import ComplexODEBoundedReward

env = ComplexODEBoundedReward([1, 1], 1)

# sample a random action
action = env.action_space.sample()
# step the environment with the sampled action
state, reward, done, info = env.step(action)
# print the state, reward, done and info
print("State: ", state)
print("Reward: ", reward)
print("Done: ", done)