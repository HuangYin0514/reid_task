import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network
import utils
from torchdiffeq import odeint_adjoint as odeint


class ODEfunc(nn.Module):
    def __init__(self, dim=2048):
        super(ODEfunc, self).__init__()

        self.act = nn.ReLU(inplace=True)

        self.norm1 = nn.GroupNorm(min(32, dim), dim)

        self.layer2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, dim), dim)

        self.layer3 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.GroupNorm(min(32, dim), dim)

    def function(self, t, x):  # x -> (batch_size, 2048)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 1, 1)
        out = self.act(self.norm1(x))
        out = self.act(self.norm2(self.layer2(out)))
        out = self.norm3(self.layer3(out))
        out = out.reshape(batch_size, -1)
        return out  #  -> (batch_size, 1)

    def forward(self, t, coords):  # coords -> (batch_size, 2048)
        batch_size = coords.shape[0]
        dz_dt = self.function(t, coords)
        return dz_dt  # -> (batch_size, 2048)


class ODEBlock(nn.Module):

    def __init__(self, config, logger):
        super(ODEBlock, self).__init__()
        self.odefunc = ODEfunc()
        self.integration_time = torch.arange(0, 0.05, 0.02).float()

    def forward(self, x):  # x -> (batch_size, 2048)
        # Integration input
        ## time
        integration_time = self.integration_time.type_as(x)
        ## coords
        coords = x  # -> (batch_size, 2048)

        # Integration
        out = odeint(self.odefunc, coords, integration_time, method="rk4", rtol=1e-3, atol=1e-3)

        # Output
        out = out[-1, ...]

        return out  # -> (bs, 2048)


class Dynamics(nn.Module):
    def __init__(self, config, logger, **kwargs):
        super(Dynamics, self).__init__()

        self.config = config
        self.logger = logger

        self.ode_net = ODEBlock(config, logger)
        self.ode_net.apply(network.utils.weights_init_kaiming)

        self.ode_net2 = ODEBlock(config, logger)
        self.ode_net2.apply(network.utils.weights_init_kaiming)

    def forward(self, x):  # x -> (batch_size, 2048)
        batch_size = x.size(0)
        out = self.ode_net(x)
        out = self.ode_net2(out)
        return out  # x -> (batch_size, 2048)


class Feature_Extractor(nn.Module):
    def __init__(self, input_dim, output_dim, config, logger, **kwargs):
        super(Feature_Extractor, self).__init__()

        self.config = config
        self.logger = logger

        # self.gmp = network.layers.GeneralizedMeanPoolingP()
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.conv1 = nn.Conv2d(2048 * 2, 2048, kernel_size=1, stride=1, bias=False)
        # self.norm1 = nn.BatchNorm2d(2048)
        # self.act = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        out = x
        return out.view(batch_size, -1)

    # def forward(self, x):
    #     batch_size = x.size(0)
    #     x = x.view(batch_size, -1, 1, 1)
    #     m_feat = self.gmp(x)
    #     a_feat = self.gap(x)
    #     am_feat = torch.cat([m_feat, a_feat], dim=1)
    #     out = self.act(self.norm1(self.conv1(am_feat)))
    #     return out.view(batch_size, -1)


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

        feature_dim = 2048

        # Backbone
        self.backbone = Backbone()

        # Gloab module
        self.pool_layer = network.layers.GeneralizedMeanPoolingP()
        # self.pool_layer = nn.AdaptiveAvgPool2d(1)
        self.gloab_classifier = nn.Linear(feature_dim, self.num_classes, bias=False)
        self.gloab_classifier.apply(network.utils.weights_init_classifier)

        # Feature extractor module
        self.feature_extractor = Feature_Extractor(2048, feature_dim, config, logger)

        # Dynamics module
        self.dynamics = Dynamics(config, logger)

        # Bottleneck module
        self.bottleneck = nn.BatchNorm1d(feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(network.utils.weights_init_kaiming)

    def heatmap(self, x):
        return self.backbone(x)

    def forward(self, x):
        batch_size = x.size(0)

        # Backbone
        resnet_feat = self.backbone(x)  # (batch_size, 2048, 16, 8)

        # Gloab module ([N, 2048])
        gloab_feat = self.pool_layer(resnet_feat)  # (batch_size, 2048, 1, 1)
        gloab_feat = gloab_feat.view(batch_size, -1)  # (batch_size, 2048)

        # Dynamics module
        gloab_feat = self.dynamics(gloab_feat)  # (batch_size, feature_dim)

        # Feature extractor module
        gloab_feat = self.feature_extractor(gloab_feat)  # (batch_size, feature_dim)

        # Bottleneck module
        norm_gloab_feat = self.bottleneck(gloab_feat)  # (batch_size, feature_dim)

        if self.training:
            # Gloab module to classifier([batch_size, num_classes]）
            gloab_score = self.gloab_classifier(norm_gloab_feat)

            return gloab_score, gloab_feat

        else:
            return norm_gloab_feat
