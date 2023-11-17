import argparse
import numpy as np
import itertools

import gym
from gym.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import vec_env, monitor
from stable_baselines3 import PPO

import wandb
from wandb.integration.sb3 import WandbCallback

NUM_OF_PARAMS = 2
NUM_OF_ENVS = 8

"""

"""