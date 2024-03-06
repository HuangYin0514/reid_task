import torch
import torch.nn as nn
from loss_funciton.crossentropy_labelsmooth_loss import CrossEntropyLabelSmoothLoss
from loss_funciton.triplet_loss import TripletLoss_v2


class Softmax_Triplet_loss(nn.Module):
    def __init__(self, num_class, margin, epsilon, config, logger):
        super().__init__()
        self.cross_entropy = CrossEntropyLabelSmoothLoss(num_classes=num_class, epsilon=epsilon, config=config, logger=logger)
        self.triplet = TripletLoss_v2(margin=margin)

    def forward(self, score, feat, target):
        return self.cross_entropy(score, target) + self.triplet(feat, target)
