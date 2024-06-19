import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network
import utils
from torchdiffeq import odeint_adjoint as odeint


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


class Classifier2(nn.Module):
    def __init__(self, feat_dim, pid_num, config, logger, **kwargs):
        super(
            Classifier2,
            self,
        ).__init__()
        self.config = config
        self.logger = logger

        # BatchNorm
        self.BN = nn.BatchNorm1d(feat_dim)
        self.BN.bias.requires_grad_(False)
        self.BN.apply(network.utils.weights_init_kaiming)

        # Classifier
        self.classifier = nn.Linear(feat_dim, pid_num, bias=False)
        self.classifier.apply(network.utils.weights_init_classifier)

    def forward(self, features):  # # (batch_size, dim)
        bn_features = self.BN(features)  # (batch_size, dim)
        cls_score = self.classifier(bn_features)  # (batch_size, num_classes）
        return cls_score


class ODEfunc(nn.Module):
    def __init__(self, dim, config, logger, **kwargs):
        super(ODEfunc, self).__init__()
        self.config = config
        self.logger = logger

        self.act = nn.ReLU(inplace=True)

        self.norm1 = nn.GroupNorm(min(32, dim), dim)

        self.layer2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, dim), dim)

        self.layer3 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.GroupNorm(min(32, dim), dim)

    def function(self, t, x):  # x -> (batch_size, c_dim)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 1, 1)
        out = self.act(self.norm1(x))
        out = self.act(self.norm2(self.layer2(out)))
        out = self.norm3(self.layer3(out))
        out = out.reshape(batch_size, -1)
        return out  #  -> (batch_size, 1)

    def forward(self, t, coords):  # coords -> (batch_size, c_dim)
        batch_size = coords.shape[0]
        dz_dt = self.function(t, coords)
        return dz_dt  # -> (batch_size, c_dim)


class ODEBlock(nn.Module):
    def __init__(self, dim, config, logger, **kwargs):
        super(ODEBlock, self).__init__()
        self.config = config
        self.logger = logger

        self.odefunc = ODEfunc(dim=dim)
        self.integration_time = torch.arange(0, 0.5, 0.02).float()

    def forward(self, x):  # x -> (batch_size, c_dim)
        # Integration input
        ## time
        integration_time = self.integration_time.type_as(x)
        ## coords
        coords = x  # -> (batch_size, c_dim)

        # Integration
        out = odeint(self.odefunc, coords, integration_time, method="euler", rtol=1e-3, atol=1e-3)

        # Output
        out = out[-1, ...]

        return out  # -> (bs, c_dim)


class Dynamics(nn.Module):
    def __init__(self, dim, config, logger, **kwargs):
        super(Dynamics, self).__init__()

        self.config = config
        self.logger = logger
        self.ode_net = ODEBlock(dim=dim)
        self.ode_net.apply(network.utils.weights_init_kaiming)

    def forward(self, x):  # x -> (batch_size, c_dim)
        batch_size = x.size(0)
        out = self.ode_net(x)
        return out  # x -> (batch_size, c_dim)


class Multi_Granularity_Module(nn.Module):
    def __init__(self, c_dim, input_dim, output_dim, config, logger, **kwargs):
        super(Multi_Granularity_Module, self).__init__()
        self.config = config
        self.logger = logger
        self.gmp = network.layers.GeneralizedMeanPoolingP()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(c_dim * 2, c_dim, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(c_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        m_feat = self.gmp(x)
        a_feat = self.gap(x)
        am_feat = torch.cat([m_feat, a_feat], dim=1)
        out = self.act(self.norm1(self.conv1(am_feat)))
        out = out.view(batch_size, -1)
        return out


class Hierarchical_Net(nn.Module):
    def __init__(self, c_dim, h_dim, w_dim, config, logger, **kwargs):
        super(Hierarchical_Net, self).__init__()
        self.config = config
        self.logger = logger

        self.MGM = Multi_Granularity_Module(c_dim, h_dim, w_dim)
        self.dynamics = Dynamics(c_dim)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.MGM(x)
        out = self.dynamics(out)
        return out


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
        x = self.resnet_layer4(x)  # (bs, 2048, 16, 8)
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
        resnet_feat = self.backbone(x)  # (bs, 2048, 16, 8)

        # Pooling
        pool_feat = self.GAP(resnet_feat)  # (bs, 2048, 1, 1)
        pool_feat = pool_feat.squeeze()  # (bs, 2048)

        # BN
        bn_feat = self.BN(pool_feat)  # (bs, 2048)

        if self.training:
            return pool_feat, bn_feat
        else:
            return bn_feat
