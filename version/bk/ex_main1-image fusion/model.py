import torch
import torch.nn.functional as F
from torch import nn

import network
import utils
from torchdiffeq import odeint_adjoint as odeint


class Auxiliary_Module(nn.Module):
    def __init__(self, feat_dim, pid_num, config, logger, **kwargs):
        super(Auxiliary_Module, self).__init__()
        self.quantified_classifier = Common_Classifier(feat_dim, pid_num, config, logger)
        self.feat_map_classifier = Common_Classifier(feat_dim, pid_num, config, logger)


class Common_Classifier(nn.Module):
    def __init__(self, feat_dim, pid_num, config, logger, **kwargs):
        super(
            Common_Classifier,
            self,
        ).__init__()
        self.config = config
        self.logger = logger

        # Pooling
        self.GAP = network.layers.GeneralizedMeanPoolingP()

        # BatchNorm
        self.BN = nn.BatchNorm2d(feat_dim)
        self.BN.bias.requires_grad_(False)
        self.BN.apply(network.utils.weights_init_kaiming)

        # Classifier
        self.classifier = nn.Linear(feat_dim, pid_num, bias=False)
        self.classifier.apply(network.utils.weights_init_classifier)

    def forward(self, feat):  # # (batch_size, dim)
        pool_feat = self.GAP(feat)
        bn_feat = self.BN(pool_feat).squeeze()  # (batch_size, dim)
        cls_score = self.classifier(bn_feat)  # (batch_size, num_classes）
        return bn_feat, cls_score


class Classifier(nn.Module):
    def __init__(self, feat_dim, pid_num, config, logger, **kwargs):
        super(
            Classifier,
            self,
        ).__init__()
        self.config = config
        self.logger = logger

        # Classifier
        self.classifier = nn.Linear(feat_dim, pid_num, bias=False)
        self.classifier.apply(network.utils.weights_init_classifier)

    def forward(self, features):  # # (batch_size, dim)
        cls_score = self.classifier(features)  # (batch_size, num_classes）
        return cls_score


class Backbone(nn.Module):
    def __init__(self, config, logger, **kwargs):
        super(Backbone, self).__init__()
        self.config = config
        self.logger = logger

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

        x = self.resnet_layer1(x)  # (bs, 256, 96, 32)
        x = self.resnet_layer2(x)  # (bs, 512, 48, 16)
        x = self.resnet_layer3(x)  # (bs, 1024, 24, 8)
        x = self.resnet_layer4(x)
        return x


class ReidNet(nn.Module):
    def __init__(self, config, logger, **kwargs):
        super(ReidNet, self).__init__()
        self.config = config
        self.logger = logger

        # Backbone
        self.backbone = Backbone(config, logger, **kwargs)

        # Pooling
        self.GAP = network.layers.GeneralizedMeanPoolingP()

        # BN
        self.BN = nn.BatchNorm1d(2048)
        self.BN.bias.requires_grad_(False)
        self.BN.apply(network.utils.weights_init_kaiming)

    def heatmap(self, x):
        return self.backbone(x)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        resnet_feat = self.backbone(x)  # (batch_size, 2048, 16, 8)

        # Pooling
        pool_feat = self.GAP(resnet_feat)  # (bs, 2048, 1, 1)
        pool_feat = pool_feat.squeeze()  # (bs, 2048)

        # BN
        bn_feat = self.BN(pool_feat)  # (bs, 2048)

        if self.training:
            return resnet_feat, pool_feat, bn_feat
        else:
            return bn_feat
