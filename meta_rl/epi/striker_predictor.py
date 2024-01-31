import sys
sys.path.append(".")
sys.path.append("...")
from scripts.iida.predictor import Predictor
from .striker_avg import StrikerAvgEnv

import numpy as np
import stable_baselines3
import gym
import torch
from meta_rl.definitions import RESULTS_DIR
"""
Uses trained predictor model to give latent reprentation of an environment.
"""

LATENT_SIZE = 8

class StrikerPredictorEnv(StrikerAvgEnv):
    def __init__(self,eval_mode = False, eval_scale=None):
        self.eval_mode = eval_mode
        self.eval_scale = eval_scale
        self.latent = np.zeros(LATENT_SIZE, dtype=float) 
        super(StrikerPredictorEnv, self).__init__()
        # Load the generator model
        self.generator_model = stable_baselines3.PPO.load(RESULTS_DIR / "iida/ppo_generator.zip")
        # Load the model
        self.predictor_model = Predictor.load_from_checkpoint(checkpoint_path=RESULTS_DIR / "iida/epoch=49-step=78150.ckpt")

        


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
    #print(torch.unsqueeze(torch.Tensor(s_[:-1]),1).shape)
    traj_dict = {
        "s_context": torch.unsqueeze(torch.Tensor(s_[:-1]),0),
        "a_context": torch.unsqueeze(torch.Tensor(a_[:-1]),0),
        "sp_context": torch.unsqueeze(torch.Tensor(s_[1:]),0),
    }
    # Get the latent representation
    latents_mean, latents_std = predictor_model.encoder(traj_dict).squeeze().detach().numpy()
    #print("latent", latent.shape)

    return np.concatenate([latents_mean, latents_std], axis=0)