"""
Modified version of stable_baselines3.common.buffers.ReplayBuffer
Buffer now stores the context id for each transition
And can return context id and context for each element of the batch
"""

from typing import NamedTuple, Optional
import time
import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
import random

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    context_ids: th.Tensor


class ReplayBuffer(ReplayBuffer):
    """
    Modified ReplayBuffer. Stores the context id for each transition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.hashmap = {} # key: context_id, value: list of positions in the buffer

    def add(self, obs, next_obs, action, reward, done, infos):
        """
        Add a new transition to the buffer.
        """
        context_id = int(infos["context_id"])

        #old_context_id = int(self.context_ids[self.pos])
        # Store the context id
        self.context_ids[self.pos] = context_id
        super().add(obs, next_obs, action, reward, done, infos)
        # Store the transition in the hashmap
        if context_id in self.hashmap:
            # Remove previous value associated with context_id
            self.hashmap[context_id] = [pos for pos in self.hashmap[context_id] if pos != self.pos]
            self.hashmap[context_id].append(self.pos)
        else:
            self.hashmap[context_id] = [self.pos]
                
            
    def sample_from_context(self, 
                            context_ids:np.ndarray, 
                            nb_input_transitions:int
                            ):
        """
        Sample nb_input_transitions transitions from each context in context_ids.
        returns a tensor of shape (len(context_ids), nb_input_transitions, transitions_dim)
        """
        transitions_dim = int(
            2 * np.array(self.obs_shape).prod()
            + np.array(self.action_space.shape).prod()
        )
        contexts = np.empty([len(context_ids), nb_input_transitions, int(transitions_dim)])

        for i, context_id in enumerate(context_ids):
            #print(self.hashmap[context_id])
            sampled_ids = np.random.choice(self.hashmap[context_id], nb_input_transitions)

            c = np.concatenate(
                    (
                        self._normalize_obs(
                            self.observations[sampled_ids, :],
                            None,
                        ),
                        self.actions[sampled_ids, :],
                        self._normalize_obs(
                            self.next_observations[sampled_ids, :],
                            None,
                        ),
                    ),
                    axis=-1,
                )
            # convert to tensor
            contexts[i] = c.squeeze(1)
    
           
        return th.as_tensor(contexts, dtype=th.float32)


        
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
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