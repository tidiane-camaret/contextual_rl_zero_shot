import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from model import TrajDataset, Predictor
import stable_baselines3
generator_model = stable_baselines3.PPO.load("scripts/iida/ppo_generator")

