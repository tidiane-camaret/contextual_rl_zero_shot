### The replay buffer needs to output a batch of data for the agent to train on.
### The data is a tuple of (obs, action, next_obs, done, reward)

import torch
import torch.nn as nn


# define the context encoder
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
                modules.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            modules.append(activation())
            modules.append(nn.Linear(hidden_sizes[-1], d_out))

            self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class ContextEncoder(nn.Module):
    """
    Encodes a context of shape [B, context_length, context_dim] into a latent vector.
    """

    def __init__(self, d_in, d_out, hidden_sizes, activation=nn.ReLU):
        super(ContextEncoder, self).__init__()
        self.model = FeedForward(d_in, d_out, hidden_sizes, activation)

    def forward(self, x):
        # flatten x to [B * context_length, context_dim]
        # pass x through the model
        latents = self.model(x.view(-1, x.shape[-1]))
        # reshape latents back to [B, context_length, d_out]
        latents = latents.view(x.shape[0], x.shape[1], -1)

        latents_mean = torch.mean(latents, dim=1)
        latents_std = torch.std(latents, dim=1)
        return latents_mean, latents_std
