import os

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from utils import to_pickle


class Recorder:
    def __init__(self, config, logger, **kwargs):
        super(Recorder, self).__init__()

        self.config = config
        self.logger = logger

        self.train_epochs_list = []
        self.train_loss_list = []

        self.val_epochs_list = []
        self.val_CMC_list = []
        self.val_mAP_list = []

    def save(self):
        stats = {}

        stats["train_epochs_list"] = self.train_epochs_list
        stats["train_loss_list"] = self.train_loss_list

        stats["val_epochs_list"] = self.val_epochs_list
        stats["val_CMC_list"] = self.val_CMC_list
        stats["val_mAP_list"] = self.val_mAP_list

        filename = f"recorder.pkl"
        path = os.path.join(self.config.outputs_path, filename)
        to_pickle(stats, path)
