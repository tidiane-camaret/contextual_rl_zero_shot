import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from meta_rl.definitions import RESULTS_DIR

class TrajDataset(Dataset):
    """
    Returns a list of (s, a, s') tuples of the same context.
    """
    def __init__(self, 
                 traj_dict, 
                 context_size=128, # 
                 ds_size=100_000 #
                 ):
        self.traj_dict = traj_dict
        
        self.context_size = context_size
        self.ds_size = ds_size

    def __len__(self):
        return self.ds_size

    def __getitem__(self, idx):
        # pick a context at random
        context = np.random.choice(list(self.traj_dict.keys()))
        # pick context_size number of tuples from the context
        idxs = np.random.choice(len(self.traj_dict[context]), self.context_size+1, replace=False)
        
        return {"s": torch.tensor(np.array(self.traj_dict[context][idxs[-1]][0])).type(torch.float32),
                "a": torch.tensor(np.array(self.traj_dict[context][idxs[-1]][1])).type(torch.float32),
                "sp": torch.tensor(np.array(self.traj_dict[context][idxs[-1]][2])).type(torch.float32),
                "s_context": torch.tensor(np.array([self.traj_dict[context][i][0] for i in idxs[:-1]])).type(torch.float32),
                "a_context": torch.tensor(np.array([self.traj_dict[context][i][1] for i in idxs[:-1]])).type(torch.float32),
                "sp_context": torch.tensor(np.array([self.traj_dict[context][i][2] for i in idxs[:-1]])).type(torch.float32),
                #TODO : optimize dataloading
                # TODO : array before saving the trajs would be faster
                "context": context
                }
## Define the predictor model
class FeedForward(nn.Module):
    def __init__(self, d_in, d_out, hidden_sizes, activation=nn.ReLU):
        super(FeedForward, self).__init__()
        if len(hidden_sizes) == 0:
            self.model = nn.Linear(d_in, d_out)
        else:
            modules = [nn.Linear(d_in, hidden_sizes[0])]
            for i in range(len(hidden_sizes) - 1):
                modules.append(activation())
                modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            modules.append(activation())    
            modules.append(nn.Linear(hidden_sizes[-1], d_out))

            self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
    
class MultipleEncoder(nn.Module):
    """
    Encodes multiple (s, a, s') tuples into a latent vector.
    """
    def __init__(self, d_in, d_out, hidden_sizes, activation=nn.ReLU):
        super(MultipleEncoder, self).__init__()
        self.model = FeedForward(d_in, d_out, hidden_sizes, activation)


    def forward(self, x):
        latents = []
        s, a, sp = x["s_context"], x["a_context"], x["sp_context"]
        for i in range(s.shape[1]):
            cat_input = torch.cat([s[:, i, :], a[:, i, :], sp[:, i, :]], dim=-1)
            latent = self.model(cat_input)
            latents.append(latent)
        latents = torch.stack(latents)
        #print("outputs", outputs.shape)
        latents_mean = torch.mean(latents, dim=0)
        latents_std = torch.std(latents, dim=0)
        #print("outputs_mean", outputs_mean.shape)
        return latents_mean, latents_std
    
class Decoder(nn.Module):
    """
    Decodes (s, a, latent) into s' 
    """
    def __init__(self, d_in, d_out, hidden_sizes, activation=nn.ReLU):
        super(Decoder, self).__init__()
        self.model = FeedForward(d_in, d_out, hidden_sizes, activation)

    def forward(self, x, latent):
        s, a = x["s"], x["a"]
        x = torch.cat([s, a, latent], dim=-1)
        return self.model(x)
    
class Predictor(pl.LightningModule):
    def __init__(self, 
                 d_obs=23, 
                 d_act=7,
                 d_latent=8, 
                 hidden_sizes=[32,32], 
                 activation=nn.ReLU,
                 lr=1e-3,
                 ):
        super().__init__()
        self.lr = lr
        self.encoder = MultipleEncoder(d_obs + d_act + d_obs, d_latent, hidden_sizes, activation)
        self.decoder = Decoder(d_obs + d_act + d_latent, d_obs, hidden_sizes, activation)
        self.loss = nn.MSELoss()

        #save latents for visualization
        self.latents = []
        self.contexts = []
        
    def forward(self, x):
        latents_mean, latents_std = self.encoder(x)
        return self.decoder(x, latents_mean), latents_mean
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        sp_hat, latent = self(batch)
        loss = self.loss(sp_hat, batch['sp'])
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sp_hat, latent = self(batch)
        loss = self.loss(sp_hat, batch['sp'])
        self.log('val_loss', loss, prog_bar=True)
        self.latents.extend(latent.detach().cpu().numpy())
        self.contexts.extend(batch['context'])
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.loss(self(batch), batch[-1][2])
        self.log('test_loss', loss)
        return loss
    
    def predict(self, tuples):
        return self(tuples)
    
    def on_validation_epoch_end(self):
        # plot the latent space
        #
        # contexts are strings in the form '[0.55 0.15]'.
        # we need to convert them to floats
        contexts = [np.fromstring(c[1:-1], sep=' ') for c in self.contexts]
        #print("contexts", contexts)
        #contexts = [c.astype(float) for c in self.contexts]
        latents = TSNE(n_components=2).fit_transform(np.array(self.latents))
        # plot 2 graphs, one for each dimension of the context
        for i in range(len(contexts[0])):
            plt.figure()
            plt.scatter(latents[:, 0], latents[:, 1], c=[c[i] for c in contexts])
            plt.colorbar()
            plt.savefig(RESULTS_DIR / "iida/latent_space_dim_"+i+".png")

        

