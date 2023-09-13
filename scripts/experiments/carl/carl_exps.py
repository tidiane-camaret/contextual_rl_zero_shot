import numpy as np
import gymnasium as gym
from carl.envs import CARLCartPole as CARLEnv
# discrete actions : CARLLunarLander, CARLCartPole
# continuous actions : CARLPendulum
from carl.context.context_space import NormalFloatContextFeature, UniformFloatContextFeature
from carl.context.sampler import ContextSampler
from carl_wrapper import context_wrapper
context_name = "gravity"
l,u = 0.002, 0.2
context_distributions = [UniformFloatContextFeature(context_name, l, u)]
context_sampler = ContextSampler(
                    context_distributions=context_distributions,
                    context_space=CARLEnv.get_context_space(),
                    seed=0
                )

contexts = context_sampler.sample_contexts(n_contexts=5)


CARLEnv = context_wrapper(CARLEnv, context_name = context_name, concat_context = True)
CARLEnv.render_mode = "human"
env = CARLEnv(
        # You can play with different gravity values here
        contexts=contexts,
        obs_context_as_dict=True,
        hide_context = True,
        )

def make_env(env_id, seed, idx, capture_video, run_name, hide_context):
    def thunk():
        mu, rel_sigma = 10, 0.5
        context_distributions = [NormalFloatContextFeature("gravity", mu, rel_sigma*mu)]
        context_sampler = ContextSampler(
                            context_distributions=context_distributions,
                            context_space=CARLEnv.get_context_space(),
                            seed=seed,
                        )
        
        contexts = context_sampler.sample_contexts(n_contexts=100)
        # contexts={0: CARLEnv.get_default_context()}
        env = CARLEnv(
        # You can play with different gravity values here
        contexts=contexts,
        #obs_context_as_dict=False,
        hide_context = hide_context,
        )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed) 

        return env

    return thunk

env = gym.vector.SyncVectorEnv([make_env("CARLPendulum-v1", 0, i, False, "test", False) for i in range(4)])

"""
print(env.observation_space)
print(env.single_observation_space)
"""


# render the environment

env.reset()
#env.render()

for i in range(10):
    #print(i)
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    #env.render()
    print("context_id", info["context_id"]) 
    #if done:
    #    env.reset()

