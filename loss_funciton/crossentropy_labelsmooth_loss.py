import torch
from torch import nn


class CrossEntropyLabelSmoothLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
            num_classes (int): number of classes.
            epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, config=None, logger=None):
        super(CrossEntropyLabelSmoothLoss, self).__init__()
        self.config = config
        self.logger = logger

        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
                inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
                targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        # targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        # targets = targets.to(self.config.device)
        targets = torch.zeros(log_probs.size(), device=self.config.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
