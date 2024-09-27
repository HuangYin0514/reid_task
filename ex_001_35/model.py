import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class Integrate_feats_module(nn.Module):
    def __init__(self, config, logger):
        super(Integrate_feats_module, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, feats, pids, backbone_cls_score, num_same_id=4):
        bs, c, h, w = feats.size(0), feats.size(1), feats.size(2), feats.size(3)
        chunk_size = int(bs / num_same_id)
        weights = backbone_cls_score[torch.arange(bs), pids].view(chunk_size, 4)  # (chunk_size, 4)
        weights_norm = torch.softmax(weights, dim=1)
        ids_feats = feats.view(chunk_size, num_same_id, c, h, w)  # (chunk_size, 4, c, h, w)
        integrate_feats = torch.einsum("bx,bxchw->bchw", weights_norm, ids_feats)  # (chunk_size, c, h, w)
        integrate_pids = pids[::num_same_id]
        return integrate_feats, integrate_pids


class Hierarchical_aggregation(nn.Module):
    def __init__(self, num_classes, config, logger, **kwargs):
        super(
            Hierarchical_aggregation,
            self,
        ).__init__()

        self.config = config
        self.logger = logger

        self.pool_p1 = nn.MaxPool2d(kernel_size=(4, 4))
        self.pool_p2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool_p3 = nn.MaxPool2d(kernel_size=(1, 1))

        self.integrate_feats_module = Integrate_feats_module(config, logger)

        self.reduction_p1 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=True), nn.BatchNorm2d(256), nn.ReLU())
        self.reduction_p2 = nn.Sequential(nn.Conv2d(768, 768, 1, bias=True), nn.BatchNorm2d(768), nn.ReLU())
        self.reduction_p3 = nn.Sequential(nn.Conv2d(1792, 1792, 1, bias=True), nn.BatchNorm2d(1792), nn.ReLU())

        self.fc_1 = Auxiliary_classifier_head(256, num_classes, config, logger)
        self.fc_2 = Auxiliary_classifier_head(768, num_classes, config, logger)
        self.fc_3 = Auxiliary_classifier_head(1792, num_classes, config, logger)

    def forward(
        self,
        x1,
        x2,
        x3,
        backbone_cls_score,
        pids,
    ):  # (batch_size, dim)
        pool_p1 = self.pool_p1(x1)
        # integrate_feats_p1, integrate_pids = self.integrate_feats_module(pool_p1, pids, backbone_cls_score)
        # p1 = self.reduction_p1(integrate_feats_p1).squeeze(dim=3).squeeze(dim=2)
        p1 = self.reduction_p1(pool_p1).squeeze(dim=3).squeeze(dim=2)

        pool_p2 = self.pool_p2(x2)
        # integrate_feats_p2, integrate_pids = self.integrate_feats_module(pool_p2, pids, backbone_cls_score)
        # cat_p2 = torch.cat([integrate_feats_p2, p1], dim=1)
        cat_p2 = torch.cat([pool_p2, p1], dim=1)
        p2 = self.reduction_p2(cat_p2).squeeze(dim=3).squeeze(dim=2)

        pool_p3 = self.pool_p3(x3)
        # integrate_feats_p3, integrate_pids = self.integrate_feats_module(pool_p3, pids, backbone_cls_score)
        # cat_p3 = torch.cat([integrate_feats_p3, p2], dim=1)
        cat_p3 = torch.cat([pool_p3, p2], dim=1)
        p3 = self.reduction_p3(cat_p3).squeeze(dim=3).squeeze(dim=2)

        fc_1_score = self.fc_1(p1)
        fc_2_score = self.fc_2(p2)
        fc_3_score = self.fc_3(p3)

        return fc_1_score, fc_2_score, fc_3_score, pids


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
        pool_feat = self.pool_layer(feat)  # (batch_size, 2048, 1, 1)
        pool_feat = pool_feat.view(bs, -1)  # (batch_size, 2048)
        # BN
        bn_feat = self.BN(pool_feat)  # (batch_size, 2048)
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
        self.hierarchical_aggregation = Hierarchical_aggregation(num_classes, config, logger)

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
