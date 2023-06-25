import numpy as np

import stable_baselines3

from ...scripts.iida.model import Predictor
from .striker_avg import StrikerAvgEnv
from ...scripts.iida.predictor_inference import get_latent_representation

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
            self.scale,
        ])
