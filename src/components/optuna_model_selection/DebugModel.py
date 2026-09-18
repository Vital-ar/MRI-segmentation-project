import torch.nn as nn

import torch

class Debug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    