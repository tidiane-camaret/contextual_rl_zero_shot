### The replay buffer needs to output a batch of data for the agent to train on.
### The data is a tuple of (obs, action, next_obs, done, reward)
from typing import NamedTuple, Optional
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


from carl.context.context_space import NormalFloatContextFeature, UniformFloatContextFeature
from carl.context.sampler import ContextSampler
from carl.envs import CARLCartPole as CARLEnv

from carl_wrapper import context_wrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0
num_envs = 4
buffer_size = 1000
context_name = "gravity"
hide_context = False
total_timesteps = 1000
batch_size = 16
# wrap the env so it returns the observation as an array instead of a dict
CARLEnv = context_wrapper(CARLEnv, 
                    context_name = context_name, 
                    concat_context = not hide_context)

def make_env(seed, context_name = "gravity"):
    def thunk():
        context_default = CARLEnv.get_default_context()[context_name]
        
        #mu, rel_sigma = 10, 5
        #context_distributions = [NormalFloatContextFeature(context_name, mu, rel_sigma*mu)]            
        l, u = context_default * 0.2, context_default * 2.2
        context_distributions = [UniformFloatContextFeature(context_name, l, u)]
        
        context_sampler = ContextSampler(
                            context_distributions=context_distributions,
                            context_space=CARLEnv.get_context_space(),
                            seed=seed,
                        )
        
        contexts = context_sampler.sample_contexts(n_contexts=100)
        env = CARLEnv(
        # You can play with different gravity values here
        contexts=contexts,
        obs_context_as_dict=True,
        )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed) 

        return env

    return thunk
envs = gym.vector.SyncVectorEnv(
        [make_env(seed + i, context_name=context_name) for i in range(num_envs)]
    )

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    context_ids: th.Tensor

class ReplayBuffer(ReplayBuffer):
    """
    Modified replaybuffer. Stores the context id for each transition.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

    def add(self, obs, next_obs, action, reward, done, infos):
        """
        Add a new transition to the buffer.
        """
        super().add(obs, next_obs, action, reward, done, infos)
        self.context_ids[self.pos] = np.array(infos["context_id"]).copy()

    def sample(self, batch_size: int, context_id: int = None, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        #TODO: sample based on context id
        """
        # Indices where context_id is equal to the given context_id
        if context_id is not None:
            context_inds = np.where(self.context_ids == context_id)
            # Sample from the context_inds
            ids = np.random.randint(0, context_inds[0].shape[0], size=batch_size)
            batch_inds = (context_inds[0][ids], context_inds[1][ids])
            # since context_inds is a 2d matrix (buffer_size, n_envs), we need to get the env indices as well
            return self._get_samples(batch_inds=batch_inds[0], env_indices=batch_inds[1], env=env)
        # If no context_id is given, sample randomly
        else:
            batch_inds = np.random.randint(0, self.buffer_size, size=batch_size)
            return self._get_samples(batch_inds, env=env)
    

    def _get_samples(self, batch_inds: np.ndarray, env_indices: np.ndarray = None, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        if env_indices is None:
            env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.context_ids[batch_inds, env_indices]
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))



rb = ReplayBuffer(
    buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    handle_timeout_termination=False,
    n_envs=num_envs,

)


# reset the environments
obs, _ = envs.reset(seed=seed)

# run the environments for timesteps and store the transitions
for global_step in range(total_timesteps):
    # pick random actions for each env
    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    # step the environments with the chosen actions
    next_obs, rewards, terminated, truncated, infos = envs.step(actions)
    # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`

    real_next_obs = next_obs.copy()
    for idx, d in enumerate(truncated):
        if d:
            real_next_obs[idx] = infos["final_observation"][idx]
    rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

# sample some data from the replay buffer
data = rb.sample(batch_size)

print(data.context_ids)
contexts = [rb.sample(batch_size, context_id=c.item()) for c in data.context_ids] 


for context in contexts:
    print(context.context_ids)
