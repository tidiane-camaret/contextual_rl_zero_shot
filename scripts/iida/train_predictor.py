import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from model import TrajDataset, Predictor

# Load the dataset
with open('scripts/iida/traj_dict_train.pkl', 'rb') as f:
    traj_dict_train = pickle.load(f)

with open('scripts/iida/traj_dict_test.pkl', 'rb') as f:
    traj_dict_test = pickle.load(f)

# Create the datasets
train_dataset = TrajDataset(traj_dict_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TrajDataset(traj_dict_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create the model
model = Predictor(d_obs=23, d_act=7, d_latent=2, hidden_sizes=[64, 64])

# Create the trainer
wandb_logger = WandbLogger(project="meta_rl_predictor",)
                           

trainer = pl.Trainer(
    logger=wandb_logger,
    #gpus=1, 
    max_epochs=10
    )

# Train the model
trainer.fit(model, 
            train_dataloader, 
            test_dataloader,)

