"""
Uses trained predictor model to give latent reprentation of an environment.
"""
import gym
import pytorch_lightning as pl
import stable_baselines3
from model import Predictor




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