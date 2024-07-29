import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class SAMS(nn.Module):
    # https://github.com/CHENGY12/APNet/blob/main/modeling/baseline.py
    """
    Split-Attend-Merge-Stack agent
    Input an feature map with shape H*W*C, we first split the feature maps into
    multiple parts, obtain the attention map of each part, and the attention map
    for the current pyramid level is constructed by mergiing each attention map.
    """

    def __init__(self, in_channels, channels, radix=4, reduction_factor=4, norm_layer=nn.BatchNorm2d):
        super(SAMS, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.channels = channels
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=1)

    def forward(self, x):

        batch, channel = x.shape[:2]
        splited = torch.split(x, channel // self.radix, dim=1)

        gap = sum(splited)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        atten = torch.split(atten, channel // self.radix, dim=1)

        out = torch.cat([att * split for (att, split) in zip(atten, splited)], 1)
        return out.contiguous()


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class BN2d(nn.Module):
    def __init__(self, planes):
        super(BN2d, self).__init__()
        self.bottleneck2 = nn.BatchNorm2d(planes)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(network.utils.weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck2(x)


class IntegrateFeatsModule(nn.Module):
    def __init__(self, config, logger):
        super(IntegrateFeatsModule, self).__init__()
        self.config = config
        self.logger = logger

        conv11 = nn.Conv2d(2048, 1, kernel_size=1, stride=1, bias=False)
        bn = nn.BatchNorm2d(1)
        act = nn.ReLU(inplace=True)
        self.weights_layer = nn.Sequential(conv11, bn, act)

    def forward(self, feats, pids, num_same_id=4):
        bs = feats.size(0)
        c, h, w = feats.size(1), feats.size(2), feats.size(3)
        chunk_size = int(bs / num_same_id)  # 15

        # Weights layer
        feats_weights = self.weights_layer(feats)
        weighted_feats = feats_weights * feats

        feats_reshaped = weighted_feats.view(chunk_size, num_same_id, c, h, w)  # 将 feats 重塑为 (chunk_size, 4, 1, h, w)
        integrate_feats = 1 / num_same_id * torch.sum(feats_reshaped, dim=1).cuda()  # 计算 integrate_feats，shape 变为 (chunk_size, c, h, w)
        integrate_pids = pids[::num_same_id]  # 直接从 pids 中获取 integrate_pids

        return integrate_feats, integrate_pids


class Auxiliary_classifier_head(nn.Module):
    def __init__(self, feat_dim, num_classes, config, logger, **kwargs):
        super(
            Auxiliary_classifier_head,
            self,
        ).__init__()
        self.config = config
        self.logger = logger

        # Pooling
        self.pool_layer = nn.AdaptiveAvgPool2d(1)

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
        return pool_feat, bn_feat, cls_score


class Classifier_head(nn.Module):
    def __init__(self, feat_dim, num_classes, config, logger, **kwargs):
        super(
            Classifier_head,
            self,
        ).__init__()
        self.config = config
        self.logger = logger

        # Pooling
        self.pool_layer = nn.AdaptiveAvgPool2d(1)

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
        return pool_feat, bn_feat, cls_score


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        # Backbone
        resnet = network.backbones.resnet50(pretrained=True)
        # resnet = network.backbones.resnet50_ibn_a(pretrained=True)

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

        # Auxiliary classifier
        self.auxiliary_classifier_head = Auxiliary_classifier_head(2048, num_classes, config, logger)

        # Integrat Feats Module
        self.integrateFeatsModule = IntegrateFeatsModule(config, logger)

        # Attention
        self.level = 2
        self.att5 = SELayer(2048, 256)
        self.att_s5 = SAMS(2048, int(2048 / self.level), radix=self.level)
        self.BN5 = BN2d(2048)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        resnet_feats = self.backbone(x)  # (bs, 2048, 16, 8)
        y = self.att5(resnet_feats)
        resnet_feats = resnet_feats * y.expand_as(resnet_feats)

        # Classifier head
        G_pool_feats, G_bn_feats, G_cls_score = self.classifier_head(resnet_feats)

        if self.training:
            return G_cls_score, G_pool_feats, G_bn_feats, resnet_feats
        else:
            return G_bn_feats
