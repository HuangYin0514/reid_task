import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from utils import util_torchtool


class Loss_Record:
    def __init__(self, **kwargs):
        super(Loss_Record, self).__init__()
    