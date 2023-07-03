"""
Use the rllab library : 

export PYTHONPATH=/home/tidiane/dev/automl/rescale/rllab/:$PYTHONPATH
./scripts/setup_linux.sh
source activate rllab3

"""

import os
import gym
from stable_baselines3 import PPO
from meta_rl.interaction_policy.interaction_policy import EPI_PPO
from meta_rl.interaction_policy.prediction_model import PredictionModel
"""
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
import pickle
import rllab.misc.logger as logger
import os.path as osp
import datetime
import tensorflow as tf
import argparse
from sandbox.rocky.tf.algos.trpo import TRPO as BasicTRPO
"""

env = gym.make("CartPole-v1", )
pred_model = PredictionModel(dir=os.getcwd() + "/data")
model = EPI_PPO(policy="MlpPolicy", pred_model=pred_model, env=env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")