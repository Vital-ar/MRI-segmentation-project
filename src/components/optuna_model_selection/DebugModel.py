import torch.nn as nn

import torch

class Debug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return self.conv(x)
    