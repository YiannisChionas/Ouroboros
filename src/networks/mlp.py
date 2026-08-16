import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, num_layers=2):
        super().__init__()
        hidden_dim = hidden_dim or 4 * in_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, in_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
