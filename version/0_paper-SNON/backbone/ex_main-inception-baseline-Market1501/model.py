import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network

class Resnet50_Baseline(nn.Module):
    def __init__(self):
        super(Resnet50_Baseline, self).__init__()

        # Backbone
        # resnet = network.backbones.resnet50(pretrained=True)
        resnet = network.backbones.inceptionv4(num_classes=0, pretrained=True)

        # Modifiy backbone
        self.features = resnet.features

    def forward(self, x):
        x = self.features(x)
        return x



class ReidNet(nn.Module):
    def __init__(self, num_classes, config, logger, **kwargs):
        super(ReidNet, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.logger = logger

        # Backbone
        self.backbone = Resnet50_Baseline()

        # Gloab module
        self.gloab_avgpool = nn.AdaptiveAvgPool2d(1)
        self.gloab_bottleneck = nn.BatchNorm1d(1536)
        self.gloab_bottleneck.bias.requires_grad_(False)
        self.gloab_classifier = nn.Linear(1536, self.num_classes, bias=False)
        self.gloab_bottleneck.apply(network.utils.weights_init_kaiming)
        self.gloab_classifier.apply(network.utils.weights_init_classifier)

    def heatmap(self, x):
        return self.backbone(x)

    def forward(self, x):
        batch_size = x.size(0)

        # Backbone
        resnet_feat = self.backbone(x)  # (batch_size, 1536, 16, 8)

        # Gloab module ([N, 1536])
        gloab_feat = self.gloab_avgpool(resnet_feat)  # (batch_size, 1536, 1, 1)
        gloab_feat = gloab_feat.view(batch_size, -1)  # (batch_size, 1536)
        norm_gloab_feat = self.gloab_bottleneck(gloab_feat)  # (batch_size, 1536)

        if self.training:
            # Gloab module to classifier([N, num_classes]）
            gloab_score = self.gloab_classifier(norm_gloab_feat)

            return gloab_score, gloab_feat

        else:
            return norm_gloab_feat
