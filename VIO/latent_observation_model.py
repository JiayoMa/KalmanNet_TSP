import torch
import torch.nn as nn


class LatentObservationModel(nn.Module):

    def __init__(self, state_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        return self.net(x)
