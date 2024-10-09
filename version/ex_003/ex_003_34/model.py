import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


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
    def __init__(self, num_classes, config, logger, init_param=True, **kwargs):
        super(ReidNet, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.logger = logger

        # Backbone
        self.backbone = Backbone()

        # Global
        global_feat_dim = 2048
        ## Pooling
        self.backbone_pool_layer = network.layers.GeneralizedMeanPoolingP()
        ## BatchNorm
        self.backbone_BN = nn.BatchNorm1d(global_feat_dim)
        if init_param:
            self.backbone_BN.bias.requires_grad_(False)
            self.backbone_BN.apply(network.utils.weights_init_kaiming)
        ## Classifier head
        self.backbone_classifier = nn.Linear(global_feat_dim, num_classes, bias=False)
        if init_param:
            self.backbone_classifier.apply(network.utils.weights_init_classifier)

        # PCB
        part_feat_dim = 256
        self.num_parts = 4
        self.part_avgpool = network.layers.GeneralizedMeanPoolingP(output_size=(self.num_parts, 1))
        self.part_local_conv_list = nn.ModuleList()
        for _ in range(self.num_parts):
            local_conv = nn.Sequential(
                nn.Conv1d(global_feat_dim, part_feat_dim, kernel_size=1), nn.BatchNorm1d(part_feat_dim), nn.ReLU(inplace=True)
            )
            self.part_local_conv_list.append(local_conv)

        self.part_classifier_head_list = nn.ModuleList()
        for _ in range(self.num_parts):
            fc = nn.Linear(part_feat_dim, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.part_classifier_head_list.append(fc)

    def forward(self, x):
        bs = x.size(0)

        # Resnet
        resnet_feats = self.backbone(x)  # (bs, 2048, 16, 8)

        # backbone
        backbone_pool_feats = self.backbone_pool_layer(resnet_feats).view(bs, -1)  # (batch_size, 2048, 1, 1) -> (batch_size, 2048)
        backbone_bn_feats = self.backbone_BN(backbone_pool_feats)  # (batch_size, 2048)

        # PCB
        feat = self.part_avgpool(resnet_feats)  # (batch_size, 2048, 6, 1)
        part_feats_list = []
        for i in range(self.num_parts):
            stripe_feat_H = self.part_local_conv_list[i](feat[:, :, i, :])  # (batch_size, 256, 1)
            part_feats_list.append(stripe_feat_H)

        if self.training:
            backbone_cls_score = self.backbone_classifier(backbone_bn_feats)
            part_score_list = [self.part_classifier_head_list[i](part_feats_list[i].view(bs, -1)) for i in range(self.num_parts)]
            return backbone_cls_score, backbone_pool_feats, part_score_list
        else:
            return backbone_bn_feats
