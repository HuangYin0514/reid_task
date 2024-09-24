import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network
from torchdiffeq import odeint_adjoint as odeint


class Feats_Fusion_Module(nn.Module):
    def __init__(self, config, logger):
        super(Feats_Fusion_Module, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, feats1, feats2):
        bs = feats1.size(0)
        alpha = 0.01
        fusion_feats = (1 - alpha) * feats1 + alpha * feats2
        return fusion_feats


class ODEfunc(nn.Module):
    def __init__(self, dim=256):
        super(ODEfunc, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.norm1 = nn.GroupNorm(min(32, dim), dim)

        # self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm2 = nn.GroupNorm(min(32, dim), dim)

        # self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm3 = nn.GroupNorm(min(32, dim), dim)

    def forward(self, t, x):
        out = self.relu(self.norm1(x))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return out


class ODEBlock(nn.Module):

    def __init__(self, config, logger):
        super(ODEBlock, self).__init__()
        self.config = config
        self.logger = logger

        self.odefunc = ODEfunc()
        # self.integration_time = torch.tensor([0, 0.01, 0.02, 0.03]).float()
        # self.integration_time = torch.tensor([0, 0.01]).float()
        self.integration_time = torch.tensor([0, 0.02, 0.04]).float()

    def forward(self, x):
        integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, integration_time, method="euler", rtol=1e-3, atol=1e-3)
        return out[-1]


class Reminder_feats_module(nn.Module):
    def __init__(self, config, logger):
        super(Reminder_feats_module, self).__init__()
        self.config = config
        self.logger = logger

        conv11 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)
        bn = nn.BatchNorm2d(256)
        act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.down_layers = nn.Sequential(conv11, bn, act)

        conv11_1 = nn.Conv2d(256, 2048, kernel_size=1, stride=1, bias=False)
        bn_1 = nn.BatchNorm2d(2048)
        act_1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.up_layers = nn.Sequential(conv11_1, bn_1, act_1)
        self.ode_net = ODEBlock(config, logger)

    def forward(self, feats):
        bs = feats.size(0)
        feats = self.down_layers(feats)
        reminder_feats = self.ode_net(feats)
        reminder_feats = self.up_layers(reminder_feats)
        return reminder_feats


class Integrate_feats_module(nn.Module):
    def __init__(self, classifier_head, config, logger):
        super(Integrate_feats_module, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, feats, pids, backbone_cls_score, num_same_id=4):
        bs = feats.size(0)
        c, h, w = feats.size(1), feats.size(2), feats.size(3)
        chunk_size = int(bs / num_same_id)  # 15

        # Ids cluster
        ids_feats = feats.view(chunk_size, num_same_id, c, h, w)  # (chunk_size, 4, c, h, w)

        # Weights
        # weights = torch.ones(15, 4, device=self.config.device)  # (chunk_size, 4)
        # print("backbone_cls_score", backbone_cls_score[torch.arange(bs), pids].shape)
        weights = backbone_cls_score[torch.arange(bs), pids].view(chunk_size, 4)  # (chunk_size, 4)
        # print("weights", weights)
        weights_norm = torch.softmax(weights, dim=1)
        # print("weights_norm", weights_norm)

        # Integrate
        integrate_feats = torch.einsum("bx,bxchw->bchw", weights_norm, ids_feats)  # (chunk_size, c, h, w)
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
        self.integrate_feats_module = Integrate_feats_module(self.classifier_head, config, logger)

        # Reminder Feats Module
        self.reminder_feats_module = Reminder_feats_module(config, logger)

        # Feats Fusion Module
        self.feats_Fusion_Module = Feats_Fusion_Module(config, logger)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        resnet_feats = self.backbone(x)  # (bs, 2048, 16, 8)
        reminder_feats = self.reminder_feats_module(resnet_feats)  # (bs, 2048, 16, 8)
        fusion_feats = self.feats_Fusion_Module(resnet_feats, reminder_feats)  # (bs, 2048, 16, 8)

        # Classifier head
        backbone_pool_feats, backbone_bn_feats, backbone_cls_score = self.classifier_head(fusion_feats)

        if self.training:
            return backbone_cls_score, backbone_pool_feats, backbone_bn_feats, resnet_feats, reminder_feats, fusion_feats
        else:
            return backbone_bn_feats
