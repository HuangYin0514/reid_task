import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class Multi_granularity(nn.Module):
    def __init__(self, feat_dim, num_classes, config, logger, **kwargs):
        super(
            Multi_granularity,
            self,
        ).__init__()

        self.config = config
        self.logger = logger

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(64, 32))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(32, 16))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(16, 8))

        self.reduction_p1 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU())
        self.reduction_p2 = nn.Sequential(nn.Conv2d(512, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU())
        self.reduction_p3 = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=False), nn.BatchNorm2d(1024), nn.ReLU())

        self.fc_1 = nn.Linear(256, num_classes)
        self.fc_2 = nn.Linear(512, num_classes)
        self.fc_3 = nn.Linear(1024, num_classes)

        # self.fc_id_2048_0.apply(network.utils.weights_init_classifier)
        # self.fc_id_2048_1.apply(network.utils.weights_init_classifier)
        # self.fc_id_2048_2.apply(network.utils.weights_init_classifier)

    def forward(self, x1, x2, x3):  # (batch_size, dim)
        zg_p1 = self.maxpool_zg_p1(x1)
        zg_p2 = self.maxpool_zg_p2(x2)
        zg_p3 = self.maxpool_zg_p3(x3)

        fg_p1 = self.reduction_p1(zg_p1).squeeze(dim=3).squeeze(dim=2)
        fg_p2 = self.reduction_p2(zg_p2).squeeze(dim=3).squeeze(dim=2)
        fg_p3 = self.reduction_p3(zg_p3).squeeze(dim=3).squeeze(dim=2)

        fc_1_score = self.fc_1(fg_p1)
        fc_2_score = self.fc_2(fg_p2)
        fc_3_score = self.fc_3(fg_p3)

        return fc_1_score, fc_2_score, fc_3_score


class Auxiliary_classifier_head(nn.Module):
    def __init__(self, feat_dim, num_classes, config, logger, **kwargs):
        super(
            Auxiliary_classifier_head,
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
        # pool_feat = self.pool_layer(feat)  # (batch_size, 2048, 1, 1)
        feat = feat.view(bs, -1)  # (batch_size, 2048)
        # BN
        bn_feat = self.BN(feat)  # (batch_size, 2048)
        # Classifier
        cls_score = self.classifier(bn_feat)  # ([N, num_classes]）
        return cls_score


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
        return pool_feat, bn_feat, cls_score


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
        x1 = x
        x = self.resnet_layer2(x)
        x2 = x
        x = self.resnet_layer3(x)
        x3 = x
        x = self.resnet_layer4(x)
        return x, x1, x2, x3


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
        self.auxiliary_classifier_head = Auxiliary_classifier_head(1280, num_classes, config, logger)

        # Multi_granularity
        self.multi_granularity = Multi_granularity(256, num_classes, config, logger)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        resnet_feats, resnet_feats_x1, resnet_feats_x2, resnet_feats_x3 = self.backbone(x)  # (bs, 2048, 16, 8)

        # Classifier head
        backbone_pool_feats, backbone_bn_feats, backbone_cls_score = self.classifier_head(resnet_feats)

        if self.training:
            return backbone_cls_score, backbone_pool_feats, backbone_bn_feats, resnet_feats, resnet_feats_x1, resnet_feats_x2, resnet_feats_x3
        else:
            return backbone_bn_feats
