"""
Modified version of stable_baselines3.common.buffers.ReplayBuffer
Buffer now stores the context id for each transition
And can return context id and context for each element of the batch
"""

from typing import NamedTuple, Optional

import numpy as np
import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    context_ids: th.Tensor
    contexts: th.Tensor


class ReplayBuffer(ReplayBuffer):
    """
    Modified ReplayBuffer. Stores the context id for each transition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

    def add(self, obs, next_obs, action, reward, done, infos):
        """
        Add a new transition to the buffer.
        """
        # Store the context id
        self.context_ids[self.pos] = np.array(infos["context_id"]).copy()
        super().add(obs, next_obs, action, reward, done, infos)
        
        

    def sample(
        self,
        batch_size: int,
        context_length: int,
        add_context: bool = False,
        context_id: int = None,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        """
        Sample a batch of transitions.
        if add_context is True, return a context tensor of shape (batch_size, transitions_dim
        """
        
        if context_id is None:
            # Sample in the buffer randomly
            sampled_idxs = np.random.randint(0, self.buffer_size, size=batch_size)
            return self._get_samples(
                batch_inds=sampled_idxs,
                env=env,
                add_context=add_context,
                context_length=context_length,
            )
        else:
            # Indices where context_id is equal to the given context_id
            context_inds = np.where(self.context_ids == context_id)
            # Sample from the context_inds
            ids = np.random.randint(0, context_inds[0].shape[0], size=batch_size)
            sampled_idxs = (context_inds[0][ids], context_inds[1][ids])
            return self._get_samples(
                batch_inds=sampled_idxs[0],
                env_indices=sampled_idxs[1],
                env=env,
                add_context=add_context,
                context_length=context_length,
            )

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_indices: np.ndarray = None,
        env: Optional[VecNormalize] = None,
        add_context: bool = False,
        context_length: int = None,
    ) -> ReplayBufferSamples:
        # Sample randomly the env idx
        if env_indices is None:
            env_indices = np.random.randint(
                0, high=self.n_envs, size=(len(batch_inds),)
            )

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        sampled_context_ids = self.context_ids[batch_inds, env_indices]

        """
        contexts is a tensor of shape (batch_size, context_length, transitions_dim)
        
        """
        if add_context:
            # transitions_dim is the sum of obs_dim, action_dim and next_obs_dim

            transitions_dim = (
                2 * np.array(self.obs_shape).prod()
                + np.array(self.action_space.shape).prod()
            )
            contexts = th.zeros(len(batch_inds), context_length, int(transitions_dim))
            for i in range(len(batch_inds)):
                # sample context_length transitions from the context_id
                context_inds = np.where(self.context_ids == sampled_context_ids[i])
                # Sample from the context_inds
                ids = np.random.randint(
                    0, context_inds[0].shape[0], size=context_length
                )
                sampled_idxs_from_context = (context_inds[0][ids], context_inds[1][ids])
                # get the (obs, action, next_obs) from the sampled_idxs_from_context

                contexts[i] = th.tensor(
                    np.concatenate(
                        (
                            self._normalize_obs(
                                self.observations[
                                    sampled_idxs_from_context[0],
                                    sampled_idxs_from_context[1],
                                    :,
                                ],
                                env,
                            ),
                            self.actions[
                                sampled_idxs_from_context[0],
                                sampled_idxs_from_context[1],
                                :,
                            ],
                            self._normalize_obs(
                                self.next_observations[
                                    sampled_idxs_from_context[0],
                                    sampled_idxs_from_context[1],
                                    :,
                                ],
                                env,
                            ),
                        ),
                        axis=1,
                    )
                )

        else:
            # empty tensor
            contexts = th.zeros(0, 0, 0)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
            sampled_context_ids,
            contexts,
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


"""
# Old version of the buffer (october 2023) doesnt return the context tensor
class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    context_ids: th.Tensor

class ReplayBuffer(ReplayBuffer):
    
    #Modified replaybuffer. Stores the context id for each transition.
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_ids = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)

    def add(self, obs, next_obs, action, reward, done, infos):
        
        #Add a new transition to the buffer.
        
        super().add(obs, next_obs, action, reward, done, infos)
        self.context_ids[self.pos] = np.array(infos["context_id"]).copy()

    def sample(self, batch_size: int, context_id: int = None, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        
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
    
"""
