import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class Classifier_head(nn.Module):
    def __init__(self, feat_dim, num_classes, config, logger, **kwargs):
        super(
            Classifier_head,
            self,
        ).__init__()
        self.config = config
        self.logger = logger

        # BatchNorm
        self.BN = nn.BatchNorm1d(feat_dim)
        self.BN.bias.requires_grad_(False)
        self.BN.apply(network.utils.weights_init_kaiming)

        # Classifier
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.classifier.apply(network.utils.weights_init_classifier)

    def forward(self, feat):  # (batch_size, dim)
        bs = feat.size(0)
        # BN
        bn_feat = self.BN(feat)  # (batch_size, 768)
        # Classifier
        cls_score = self.classifier(bn_feat)  # ([N, num_classes]）
        return bn_feat, cls_score


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        # Backbone
        # resnet = network.backbones.resnet50(pretrained=True)
        # resnet = network.backbones.resnet50_ibn_a(pretrained=True)
        self.network = network.backbones.vit_small_patch16_224_TransReID(
            img_size=[256, 128],
            sie_xishu=3.0,
            camera=0,
            view=0,
            stride_size=[16, 16],
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def forward(self, x):
        x = self.network(x)
        return x


class ReidNet(nn.Module):
    def __init__(self, num_classes, config, logger, **kwargs):
        super(ReidNet, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.logger = logger

        # Backbone
        self.backbone = Backbone()

        # Classifier head
        self.classifier_head = Classifier_head(768, num_classes, config, logger)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        backbone_feats = self.backbone(x)  # (bs, 768)

        # Gloab
        backbone_bn_feats, backbone_cls_score = self.classifier_head(backbone_feats)

        if self.training:
            return backbone_cls_score, backbone_bn_feats
        else:
            return backbone_bn_feats
