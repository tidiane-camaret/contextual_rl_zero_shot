# A wrapper for the CARL Env classes
# makes the environment return the observation as an array instead of a dict
# also gives the option to concatenate the context to the observation


import numpy as np
from gymnasium.spaces import Box


def context_wrapper(cls, context_name="gravity", concat_context=True):
    """
    Wrapping function
    """

    class Wrapper(cls):
        """
        Wrapper class for the CARLEnv classes
        Makes the environment return the observation as an array instead of a dict
        Also gives the option to concatenate the context to the observation
        """

        def __init__(
            self,
            concat_context=concat_context,
            context_name=context_name,
            *args,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.concat_context = concat_context
            self.context_name = context_name
            if not concat_context:
                self.observation_space = self.observation_space["obs"]
            else:
                # self.observation_space has to be a Box
                # with shape (self.observation_space.shape[0] + 1,)
                # where the last dimension is the context dimension
                # print("obs space", self.observation_space)
                # print("obs space", self.observation_space["obs"])
                # print("context space", self.observation_space["context"][context_name])
                self.observation_space = Box(
                    low=np.concatenate(
                        (
                            self.observation_space["obs"].low,
                            self.observation_space["context"][context_name].low,
                        )
                    ),
                    high=np.concatenate(
                        (
                            self.observation_space["obs"].high,
                            self.observation_space["context"][context_name].high,
                        )
                    ),
                    dtype=np.float32,
                )

        def step(self, action, *args, **kwargs):
            obs, reward, done, truncated, info = super().step(action, *args, **kwargs)
            obs = (
                self._concatenate_obs_context(obs, self.context_name)
                if self.concat_context
                else obs["obs"]
            )

            # info["context_id"] = self.context_id # add the context id at every step (used for gathering transitions from the same context)
            return (
                obs,
                reward,
                done,
                truncated,
                info,
            )

        def reset(self, *args, **kwargs):
            obs, info = super().reset(*args, **kwargs)
            obs = (
                self._concatenate_obs_context(obs, self.context_name)
                if self.concat_context
                else obs["obs"]
            )
            return obs, info

        def _concatenate_obs_context(self, obs, context_name):
            return np.append(obs["obs"], obs["context"][context_name])

    return Wrapper
