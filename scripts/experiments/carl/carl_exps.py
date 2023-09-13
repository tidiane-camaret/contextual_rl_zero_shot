import numpy as np
from carl.envs import CARLPendulum as CARLEnv
# discrete actions : CARLLunarLander 
# continuous actions : CARLPendulum
from carl.context.context_space import NormalFloatContextFeature, UniformFloatContextFeature
from carl.context.sampler import ContextSampler

context_name = "gravity"
l,u = 0.002, 0.2
context_distributions = [UniformFloatContextFeature(context_name, l, u)]
context_sampler = ContextSampler(
                    context_distributions=context_distributions,
                    context_space=CARLEnv.get_context_space(),
                    seed=0
                )

contexts = context_sampler.sample_contexts(n_contexts=5)

# wrapper for CARLEnv
# wraps the environment so that at each state, the context is added to the observation

class CARLEnvWrap(CARLEnv):
    def step(self, action):
        obs, reward, done, info, _ = super().step(action)
        #print(obs)
        context_obs = np.append(obs["obs"], obs["context"][context_name])
        return context_obs, reward, done, info, _

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        #print(obs[0])
        context_obs = np.append(obs[0]["obs"], obs[0]["context"][context_name])
        return context_obs

CARLEnvWrap.render_mode = "human"
env = CARLEnvWrap(
        # You can play with different gravity values here
        contexts=contexts,
        obs_context_as_dict=True,
        hide_context = True,
        )

env.reset()
obs, reward, done, info, _ = env.step(env.action_space.sample())

print(obs)
"""
print(obs["obs"])
print(obs["context"][context_name])

#context_obs = np.append(obs["obs"], obs["context"][context_name])
#print(context_obs)
"""

# render the environment

env.reset()
env.render()

for i in range(100):
    action = env.action_space.sample()
    obs, reward, done, info, _ = env.step(action)
    env.render()
    if done:
        env.reset()