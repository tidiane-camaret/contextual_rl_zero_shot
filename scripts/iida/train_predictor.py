import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from scripts.iida.predictor import TrajDataset, Predictor

print("Training predictor")

# Load the dataset
with open('scripts/iida/traj_dict_train.pkl', 'rb') as f:
    traj_dict_train = pickle.load(f)

with open('scripts/iida/traj_dict_test.pkl', 'rb') as f:
    traj_dict_test = pickle.load(f)

# Create the datasets
train_dataset = TrajDataset(traj_dict_train, ds_size=100_000)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

test_dataset = TrajDataset(traj_dict_test, ds_size=2_000)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Create the model
model = Predictor(d_obs=23, d_act=7, d_latent=8, hidden_sizes=[32, 32])

# Create the trainer
wandb_logger = WandbLogger(project="meta_rl_predictor",
                           save_dir = 'results',
                           log_model=True)
                           

trainer = pl.Trainer(
    logger=wandb_logger,
    #num_nodes=8, bugs when using slurm. Might need to configure that in the slurm script, TODO see https://github.com/Lightning-AI/lightning/issues/10098
    max_epochs=50
    )

# Train the model
trainer.fit(model, 
            train_dataloader, 
            test_dataloader,)

# Save the model
torch.save(model.state_dict(), "scripts/iida/predictor.ckpt")