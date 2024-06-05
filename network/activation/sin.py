import torch
from torch import nn


class SinActivation(nn.Module):
    def forward(self, input):
        return torch.sin(input)
