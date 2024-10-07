import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class Integrate_feats_module(nn.Module):
    def __init__(self, config=None, logger=None):
        super().__init__()

        self.integrate_block = nn.Sequential(
            nn.Conv2d(2048 * 4, 2048, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(2048), nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, feats, num_same_id=4):
        bs, c, h, w = feats.size(0), feats.size(1), feats.size(2), feats.size(3)
        chunk_size = int(bs / num_same_id)
        # id_view_list = []
        # for i in range(chunk_size):
        #     id_view_list.append(feats[i * num_same_id : (i + 1) * num_same_id])
        # torch.stack(id_view_list)
        chunk_feats = feats.view(chunk_size, num_same_id, c, h, w)  # (chunk_size, 4, c, h, w)
        chunk_feats = chunk_feats.permute(1, 0, 2, 3, 4)  # (4, chunk_size, c, h, w)

        integrate_feats = torch.cat([chunk_feats[0], chunk_feats[1], chunk_feats[2], chunk_feats[3]], dim=1)
        integrate_feats = self.integrate_block(integrate_feats)
        integrate_feats = self.sigmoid(integrate_feats.repeat_interleave(4, dim=0))
        integrate_feats = feats * integrate_feats

        return integrate_feats


class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y).expand_as(x)
        return y


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

        # attention
        self.attention = ECALayer(2048)

        # Classifier head
        self.classifier_head = Classifier_head(2048, num_classes, config, logger)

        # Integrate_feats_module
        self.integrate_feats_module = Integrate_feats_module(config, logger)

        # Integrate classifier_head
        self.integrate_classifier_head = Classifier_head(2048, num_classes, config, logger)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        resnet_feats = self.backbone(x)  # (bs, 2048, 16, 8)
        resnet_feats = resnet_feats * self.attention(resnet_feats)  # (bs, 2048, 16, 8)

        # Classifier head
        backbone_bn_feats, backbone_cls_score = self.classifier_head(resnet_feats)

        if self.training:
            return backbone_cls_score, backbone_bn_feats, resnet_feats
        else:
            return backbone_bn_feats
