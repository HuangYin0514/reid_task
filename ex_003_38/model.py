import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class DEE_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DEE_module, self).__init__()

        self.FC11 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(network.utils.weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(network.utils.weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(network.utils.weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC1.apply(network.utils.weights_init_kaiming)

        self.FC21 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC21.apply(network.utils.weights_init_kaiming)
        self.FC22 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC22.apply(network.utils.weights_init_kaiming)
        self.FC23 = nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC23.apply(network.utils.weights_init_kaiming)
        self.FC2 = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.FC2.apply(network.utils.weights_init_kaiming)
        self.dropout = nn.Dropout(p=0.01)

    def forward(self, x):
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x)) / 3
        x1 = self.FC1(F.relu(x1))
        x2 = (self.FC21(x) + self.FC22(x) + self.FC23(x)) / 3
        x2 = self.FC2(F.relu(x2))
        out = torch.cat((x, x1, x2), 0)
        out = self.dropout(out)
        return out


class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(
                nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(high_dim),
            )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(
                nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(self.high_dim),
            )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h

        return z


class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z


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

        self.DEE = DEE_module(1024)
        self.MFA1 = MFA_block(256, 64, 0)
        self.MFA2 = MFA_block(512, 256, 1)
        self.MFA3 = MFA_block(1024, 512, 1)

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        x_ = x
        x = self.resnet_layer1(x)
        x_ = self.MFA1(x, x_)

        x = self.resnet_layer2(x_)
        x_ = self.MFA2(x, x_)

        x = self.resnet_layer3(x_)
        x_ = self.MFA3(x, x_)

        x = self.resnet_layer4(x_)
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

    def forward(self, x):
        bs = x.size(0)

        # Resnet
        resnet_feats = self.backbone(x)  # (bs, 2048, 16, 8)

        # backbone
        backbone_pool_feats = self.backbone_pool_layer(resnet_feats).view(bs, -1)  # (batch_size, 2048, 1, 1) -> (batch_size, 2048)
        backbone_bn_feats = self.backbone_BN(backbone_pool_feats)  # (batch_size, 2048)

        if self.training:
            backbone_cls_score = self.backbone_classifier(backbone_bn_feats)

            # xp = backbone_pool_feats
            # xps = xp.view(xp.size(0), xp.size(1), 1).permute(0, 2, 1)
            # xp1, xp2, xp3 = torch.chunk(xps, 3, 0)
            # xpss = torch.cat((xp2, xp3), 1)
            # loss_ort = torch.triu(torch.bmm(xpss, xpss.permute(0, 2, 1)), diagonal=1).sum() / (xp.size(0))

            return backbone_cls_score, backbone_pool_feats
        else:
            return backbone_bn_feats
