import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from model import TrajDataset, Predictor

# Load the dataset
with open('scripts/iida/traj_dict_train.pkl', 'rb') as f:
    traj_dict_train = pickle.load(f)

print(traj_dict_train[list(traj_dict_train.keys())[0]][0])