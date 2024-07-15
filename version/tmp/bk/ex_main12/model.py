import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class SuggestFeatsModule(nn.Module):
    def __init__(self, config, logger):
        super(SuggestFeatsModule, self).__init__()
        self.config = config
        self.logger = logger

        conv11 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, bias=False)
        bn = nn.BatchNorm2d(2048)
        act = nn.ReLU(inplace=True)
        self.suggest_layer = nn.Sequential(conv11, bn, act)

    def forward(self, feats):
        bs = feats.size(0)
        c, h, w = feats.size(1), feats.size(2), feats.size(3)
        feats = feats.view(bs, c, h, w)
        # Suggest layer
        suggest_feats = self.suggest_layer(feats)
        suggest_feats = suggest_feats.view(bs, c, h, w)
        return suggest_feats


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

        # Suggest Feats Module
        self.suggestFeatsModule = SuggestFeatsModule(config, logger)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        resnet_feats = self.backbone(x)  # (bs, 2048, 16, 8)
        suggest_feats = self.suggestFeatsModule(resnet_feats)
        suggest_resnet_feats = resnet_feats + suggest_feats

        # Classifier head
        G_pool_feats, G_bn_feats, G_cls_score = self.classifier_head(suggest_resnet_feats)

        if self.training:
            return G_cls_score, G_pool_feats, G_bn_feats, resnet_feats, suggest_resnet_feats, suggest_feats
        else:
            return G_bn_feats
