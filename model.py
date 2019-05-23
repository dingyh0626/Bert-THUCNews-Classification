import torch.nn as nn


class ClfModel(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, input):
        return self.seq(input)

