import sys
sys.path.append(".")
sys.path.append("...")
from scripts.iida.model import Predictor
from .striker_avg import StrikerAvgEnv

import numpy as np
import stable_baselines3
import gym

"""
Uses trained predictor model to give latent reprentation of an environment.
"""

class StrikerPredictorEnv(StrikerAvgEnv):
    def __init__(self,eval_mode = False, eval_scale=None):
        self.eval_mode = eval_mode
        self.eval_scale = eval_scale
        super(StrikerPredictorEnv, self).__init__()
        # Load the generator model
        self.generator_model = stable_baselines3.PPO.load("scripts/iida/ppo_generator.zip")
        # Load the model
        self.predictor_model = Predictor.load_from_checkpoint("scripts/iida/predictor.ckpt")



    def reset_model(self):
        self.change_env()
        self.latent = get_latent_representation(self.scale, self.predictor_model, self.generator_model)
        return self.raw_reset_model()
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:7],
            self.sim.data.qvel.flat[:7],
            self.get_body_com("tips_arm"),
            self.get_body_com("object"),
            self.get_body_com("goal"),
            self.latent,
        ])


def get_latent_representation(scale, predictor_model, generator_model):
    """
    Returns latent representation of an environment.
    """
    # Get the context
    env = gym.make('StrikerAvg-v0', eval_mode=True, eval_scale=scale)
    # TODO : maybe call this function within the env using self ?
    obs = env.reset()
    s_, a_ = [], []
    for i in range(200):
        action, _states = generator_model.predict(obs)
        obs, reward, done, info = env.step(action)
        s_.append(obs)
        a_.append(action)
    traj_dict = {
        "s": s_[:-1],
        "a": a_,
        "sp": s_[1:],
    }
    # Get the latent representation
    latent = predictor_model.encoder(traj_dict)

    return latent