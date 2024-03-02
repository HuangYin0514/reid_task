import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from utils import util_torchtool


class Recorder:
    def __init__(self, config, logger, **kwargs):
        super(Recorder, self).__init__()

        self.running_loss = 0.0
