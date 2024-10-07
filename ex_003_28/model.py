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

        # Pooling
        self.pool_layer = network.layers.GeneralizedMeanPoolingP()

        # BatchNorm
        self.BN = nn.BatchNorm1d(feat_dim)
        self.BN.bias.requires_grad_(False)
        self.BN.apply(network.utils.weights_init_kaiming)

        # Classifier
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.classifier.apply(network.utils.weights_init_classifier)

    def forward(self, feat):  # (batch_size, dim)
        bs = feat.size(0)
        # pool
        pool_feat = self.pool_layer(feat)  # (batch_size, 2048, 1, 1)
        pool_feat = pool_feat.view(bs, -1)  # (batch_size, 2048)
        # BN
        bn_feat = self.BN(pool_feat)  # (batch_size, 2048)
        # Classifier
        cls_score = self.classifier(bn_feat)  # ([N, num_classes]）
        return bn_feat, cls_score


class Part_module(nn.Module):
    def __init__(self, num_parts, config, logger, **kwargs):
        super().__init__()
        self.num_parts = num_parts
        self.part_avgpool = nn.AdaptiveAvgPool2d((self.num_parts, 1))
        self.part_local_conv_list = nn.ModuleList()
        for _ in range(self.num_parts):
            local_conv = nn.Sequential(nn.Conv1d(2048, 256, kernel_size=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
            self.part_local_conv_list.append(local_conv)

    def forward(self, feat):
        feat = self.part_avgpool(feat)
        part_feats_list = []
        for i in range(self.num_parts):
            stripe_feat_H = self.part_local_conv_list[i](feat[:, :, i, :])
            part_feats_list.append(stripe_feat_H)

        return part_feats_list


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        # Backbone
        # resnet = network.backbones.resnet50(pretrained=True)
        resnet = network.backbones.resnet50_ibn_a(pretrained=True)

        # Modifiy backbone
        ## Modifiy the stride of last conv layer
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        ## Remove avgpool and fc layer of resnet
        self.resnet_conv1 = resnet.conv1
        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)
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
        self.classifier_head = Classifier_head(2048, num_classes, config, logger)

        # PCB
        self.num_parts = 6
        self.part_module = Part_module(self.num_parts, config, logger)

        self.part_classifier_head_list = nn.ModuleList()
        for _ in range(self.num_parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.part_classifier_head_list.append(fc)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        resnet_feats = self.backbone(x)  # (bs, 2048, 16, 8)

        # Gloab
        backbone_bn_feats, backbone_cls_score = self.classifier_head(resnet_feats)

        # PCB
        part_feats_list = self.part_module(resnet_feats)  # (bs, 6, 256)

        if self.training:
            return part_feats_list, backbone_bn_feats, backbone_cls_score
        else:
            part_feats_list = torch.cat(part_feats_list, dim=2)
            part_feats_list = F.normalize(part_feats_list, p=2, dim=1)
            part_feats = part_feats_list.view(bs, -1)

            # backbone_pool_feats, backbone_bn_feats, backbone_cls_score = self.classifier_head(resnet_feats)
            return part_feats
