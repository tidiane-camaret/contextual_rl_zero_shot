import importlib
import os

import hydra
import numpy as np

from meta_rl.algorithms.sac.sac import train_sac
from meta_rl.algorithms.sac.sac_utils import Args, make_env
from meta_rl.jrpl.carl_wrapper import context_wrapper

@hydra.main(version_base=None, config_path="../configs/", config_name="base_exp")
def main(config):
    # pprint.pprint(config)
    print("Working directory : {}".format(os.getcwd()))
        # test video recording
    import gymnasium as gym


    from carl.envs import CARLBraxAnt

    env = CARLBraxAnt()

    CARLEnv = context_wrapper(
        CARLBraxAnt,
        context_name='mass_torso',
        concat_context=False,
    )
    env = CARLEnv()
    run_name = "carl_brax_ant_hydra"
    
    env = gym.vector.SyncVectorEnv(
                [make_env(env, 0, 0, True, run_name)]
            )
    """
    run_name = "carl_brax_ant"
    env = gym.wrappers.RecordVideo(env, f"results/videos/{run_name}")
    """
    obs, info = env.reset()
    for _ in range(10):
        env.step(env.action_space.sample())
    env.close()
    

if __name__ == "__main__":
    main()