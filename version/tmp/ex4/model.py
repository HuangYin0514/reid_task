import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network
import utils
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


class SinActivation(nn.Module):

    def forward(self, input):
        return torch.sin(input)


class ODEfunc(nn.Module):
    def __init__(self, dim=4096):
        super(ODEfunc, self).__init__()

        # self.relu = nn.ReLU(inplace=True)
        self.act = SinActivation()

        self.norm_start = norm(dim)

        self.conv1 = conv1x1(dim, dim)
        self.norm1 = norm(dim)

        # self.conv2 = conv1x1(dim, dim)
        # self.norm2 = norm(dim)

        self.conv_end = conv1x1(dim, 1)
        self.norm_end = norm(1)

    def function(self, t, x):
        out = self.act(self.norm_start(x))
        out = self.act(self.norm1(self.conv1(out)))
        # out = self.act(self.norm2(self.conv2(out)))
        out = self.norm_end(self.conv_end(out))
        return out

    def forward(self, t, coords):
        """ODE model

        Args:
            t (_type_): time
            coords (_type_): (bs, 2048*2, 1, 1)

        Returns:
            dynamics value:
            q: (bs, 2048, 1, 1)
            p: (bs, 2048, 1, 1)
        """
        # coords = coords.clone().detach().requires_grad_(True)
        # coords.requires_grad = True
        with torch.enable_grad():
            coords = coords.clone().detach().requires_grad_(True)
            q, p = coords.chunk(2, dim=1)
            inp = torch.cat([q, p], dim=1)
            # q = q.clone().detach().requires_grad_(True)
            # p = p.clone().detach().requires_grad_(True)
            H = self.function(t, inp)

            dqH = utils.physics.dfx(H.sum(), q)
            dpH = utils.physics.dfx(H.sum(), p)

            dq_dt = dpH
            dp_dt = -dqH

            dz_dt = torch.cat([dq_dt, dp_dt], dim=1)
        return dz_dt


class ODEBlock(nn.Module):

    def __init__(self, config, logger):
        super(ODEBlock, self).__init__()
        self.odefunc = ODEfunc()
        # self.integration_time = torch.tensor([0, 0.01, 0.02, 0.03]).float()
        # self.integration_time = torch.tensor([0, 0.01]).float()
        # self.integration_time = torch.tensor([0, 0.02, 0.04, 0.06, 0.08, 0.10]).float()
        self.integration_time = torch.arange(0, 0.05, 0.02).float()

    def forward(self, x):
        """ODE integral

        Args:
            x (_type_): coords (bs, 2048, 1, 1)

        Returns:
            features: (bs, 2048, 1, 1)
        """
        integration_time = self.integration_time.type_as(x)
        # x = x.reshape(-1, x.shape[1])
        coords_p = torch.zeros_like(x)
        coords = torch.cat((x, coords_p), dim=1)
        out = odeint(self.odefunc, coords, integration_time, method="rk4", rtol=1e-3, atol=1e-3)
        out = out[-1, ...]
        q_out, p_out = coords.chunk(2, dim=1)
        return q_out


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

        # Gloab module
        self.pool_layer = network.layers.GeneralizedMeanPoolingP()
        self.gloab_bottleneck = nn.BatchNorm1d(2048)
        self.gloab_bottleneck.bias.requires_grad_(False)
        self.gloab_classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.gloab_bottleneck.apply(network.utils.weights_init_kaiming)
        self.gloab_classifier.apply(network.utils.weights_init_classifier)

        # ODEnet module
        self.ode_net = ODEBlock(config, logger)
        self.ode_net.apply(network.utils.weights_init_kaiming)

        self.ode_net2 = ODEBlock(config, logger)
        self.ode_net2.apply(network.utils.weights_init_kaiming)

        self.ode_net3 = ODEBlock(config, logger)
        self.ode_net3.apply(network.utils.weights_init_kaiming)

    def heatmap(self, x):
        return self.backbone(x)

    def forward(self, x):
        batch_size = x.size(0)

        # Backbone
        resnet_feat = self.backbone(x)  # (batch_size, 2048, 16, 8)

        # Gloab module ([N, 2048])
        gloab_feat = self.pool_layer(resnet_feat)  # (batch_size, 2048, 1, 1)
        
        gloab_feat = self.ode_net(gloab_feat)
        gloab_feat = self.ode_net2(gloab_feat)
        
        gloab_feat = gloab_feat.view(batch_size, -1)  # (batch_size, 2048)
        norm_gloab_feat = self.gloab_bottleneck(gloab_feat)  # (batch_size, 2048)

        if self.training:
            # Gloab module to classifier([N, num_classes]）
            gloab_score = self.gloab_classifier(norm_gloab_feat)

            return gloab_score, gloab_feat

        else:
            return norm_gloab_feat
