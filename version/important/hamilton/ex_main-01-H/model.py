import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network
import utils
from torchdiffeq import odeint_adjoint as odeint


class ODEfunc(nn.Module):
    def __init__(self, dim=2048 * 2):
        super(ODEfunc, self).__init__()

        self.act = network.activation.SinActivation()

        # self.layer1 = nn.Linear(dim, dim)
        # self.norm1 = nn.BatchNorm1d(dim)

        self.layer1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(dim)

        self.layer2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(dim)

        self.layer_end = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=False)

    def function(self, t, x):  # x -> (batch_size, 2048*2)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 1, 1)
        out = self.act(self.norm1(self.layer1(x)))
        out = self.act(self.norm2(self.layer2(x))) # BUG this input
        out = self.layer_end(out)
        out = out.reshape(batch_size, -1)
        return out  #  -> (batch_size, 1)

    def forward(self, t, coords):  # coords -> (batch_size, 2048*2)
        batch_size = coords.shape[0]

        with torch.enable_grad():
            coords = coords.clone().detach().requires_grad_(True)
            q, p = coords.chunk(2, dim=1)
            inp = torch.cat([q, p], dim=-1)

            H = self.function(t, inp)

            dqH = utils.physics.dfx(H.sum(), q)
            dpH = utils.physics.dfx(H.sum(), p)

        dq_dt = dpH
        dp_dt = -dqH

        dz_dt = torch.cat([dq_dt, dp_dt], dim=1)

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
        coords_p = torch.zeros_like(x)
        coords = torch.cat([x, coords_p], dim=1)  # -> (batch_size, 2048*2)

        # Integration
        out = odeint(self.odefunc, coords, integration_time, method="rk4", rtol=1e-3, atol=1e-3)

        # Output
        out = out[-1, ...]
        q_out, p_out = out.chunk(2, dim=1)

        return q_out  # -> (bs, 2048)


class Dynamics(nn.Module):
    def __init__(self, config, logger, **kwargs):
        super(Dynamics, self).__init__()

        self.config = config
        self.logger = logger

        self.ode_net = ODEBlock(config, logger)
        self.ode_net.apply(network.utils.weights_init_kaiming)

    def forward(self, x):  # x -> (batch_size, 2048)
        batch_size = x.size(0)
        out = self.ode_net(x)
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
