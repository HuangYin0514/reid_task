import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network
from torchdiffeq import odeint_adjoint as odeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    gn = nn.GroupNorm(min(32, dim), dim)
    # bn = nn.BatchNorm2d(dim)
    return gn


class ODEfunc(nn.Module):
    def __init__(self, dim=2048):
        super(ODEfunc, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm(dim)

        self.conv2 = conv1x1(dim, dim)
        self.norm2 = norm(dim)

        self.conv3 = conv1x1(dim, dim)
        self.norm3 = norm(dim)

    def forward(self, t, x):
        out = self.relu(self.norm1(x))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        return out


class Robustness_ODEBlock(nn.Module):

    def __init__(self, config, logger):
        super(Robustness_ODEBlock, self).__init__()
        self.odefunc = ODEfunc()
        # self.integration_time = torch.tensor([0, 0.01, 0.02, 0.03]).float()
        # self.integration_time = torch.tensor([0, 0.01]).float()
        self.integration_time = torch.tensor([0, 0.02, 0.04]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        # out = odeint(self.odefunc, x, self.integration_time, method="rk4", rtol=1e-3, atol=1e-3)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1].clone().detach(), out[-1]


class Resnet50_Baseline(nn.Module):
    def __init__(self):
        super(Resnet50_Baseline, self).__init__()

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
        self.backbone = Resnet50_Baseline()

        # Gloab module
        self.gloab_avgpool = nn.AdaptiveAvgPool2d(1)
        self.gloab_bottleneck = nn.BatchNorm1d(2048)
        self.gloab_bottleneck.bias.requires_grad_(False)
        self.gloab_classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.gloab_bottleneck.apply(network.utils.weights_init_kaiming)
        self.gloab_classifier.apply(network.utils.weights_init_classifier)

        # ODEnet module
        self.ode_net = Robustness_ODEBlock(config, logger)
        self.ode_avgpool = nn.AdaptiveAvgPool2d(1)
        self.ode_bottleneck = nn.BatchNorm1d(2048)
        self.ode_bottleneck.bias.requires_grad_(False)
        self.ode_classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.ode_bottleneck.apply(network.utils.weights_init_kaiming)
        self.ode_classifier.apply(network.utils.weights_init_classifier)

    def heatmap(self, x):
        return self.backbone(x)

    def forward(self, x):
        batch_size = x.size(0)

        # Backbone
        resnet_feat = self.backbone(x)  # (batch_size, 2048, 16, 8)

        # Gloab module ([N, 2048])
        gloab_feat = self.gloab_avgpool(resnet_feat)  # (batch_size, 2048, 1, 1)
        gloab_feat = gloab_feat.view(batch_size, -1)  # (batch_size, 2048)
        norm_gloab_feat = self.gloab_bottleneck(gloab_feat)  # (batch_size, 2048)

        # ODEnet module
        ode_feat = self.ode_avgpool(resnet_feat)  # (batch_size, 2048, 1, 1)
        ode_feat_dt, ode_feat = self.ode_net(ode_feat)
        ode_feat = ode_feat.view(batch_size, -1)  # (batch_size, 2048)
        norm_ode_feat = self.ode_bottleneck(ode_feat)  # (batch_size, 2048)

        if self.training:
            # Gloab module to classifier([N, num_classes]）
            gloab_score = self.gloab_classifier(norm_gloab_feat)

            # ODEnet module to classifier([N, num_classes]）
            ode_score = self.ode_classifier(norm_ode_feat)
            _, ode_feat_Tdt = self.ode_net(ode_feat_dt)
            ode_feat_steady_diff = gloab_feat - ode_feat_Tdt.reshape(batch_size, -1)

            return gloab_score, gloab_feat, ode_score, ode_feat, ode_feat_steady_diff

        else:
            return norm_gloab_feat
