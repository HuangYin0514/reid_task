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

    def __init__(self, dim):
        super(ODEBlock, self).__init__()
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
    def __init__(self, dim, **kwargs):
        super(Dynamics, self).__init__()

        self.ode_net = ODEBlock(dim=dim)
        self.ode_net.apply(network.utils.weights_init_kaiming)

    def forward(self, x):  # x -> (batch_size, c_dim)
        batch_size = x.size(0)
        out = self.ode_net(x)
        return out  # x -> (batch_size, c_dim)


class Multi_Granularity_Module(nn.Module):
    def __init__(self, c_dim, input_dim, output_dim, **kwargs):
        super(Multi_Granularity_Module, self).__init__()

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
    def __init__(self, c_dim, h_dim, w_dim, **kwargs):
        super(Hierarchical_Net, self).__init__()
        self.MGM = Multi_Granularity_Module(c_dim, h_dim, w_dim)
        self.dynamics = Dynamics(c_dim)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.MGM(x)
        out = self.dynamics(out)
        return out


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

        self.hierarchical_net1 = Hierarchical_Net(256, 96, 32)
        self.hierarchical_net2 = Hierarchical_Net(512, 48, 16)
        self.hierarchical_net3 = Hierarchical_Net(1024, 24, 8)

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        x = self.resnet_layer1(x)  # (bs, 256, 96, 32)
        out1 = self.hierarchical_net1(x)

        x = self.resnet_layer2(x)  # (bs, 512, 48, 16)
        out2 = self.hierarchical_net2(x)

        x = self.resnet_layer3(x)  # (bs, 1024, 24, 8)
        out3 = self.hierarchical_net3(x)

        x = self.resnet_layer4(x)
        return x, out1, out2, out3


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

        # Hierarchical module
        self.hierarchical_parts = 3
        hierarchical_parts_dim = [256, 512, 1024]
        self.hierarchical_PoolBn_list = nn.ModuleList()
        for i in range(self.hierarchical_parts):
            feat_bottleneck = nn.BatchNorm1d(hierarchical_parts_dim[i])
            feat_bottleneck.bias.requires_grad_(False)
            feat_bottleneck.apply(network.utils.weights_init_kaiming)
            temp = nn.Sequential(feat_bottleneck)
            self.hierarchical_PoolBn_list.append(temp)
        self.hierarchical_FC_list = nn.ModuleList()
        for i in range(self.hierarchical_parts):
            feat_classifier = nn.Linear(hierarchical_parts_dim[i], num_classes, bias=False)
            feat_classifier.apply(network.utils.weights_init_classifier)
            temp = nn.Sequential(feat_classifier)
            self.hierarchical_FC_list.append(temp)

    def heatmap(self, x):
        return self.backbone(x)

    def forward(self, x):
        batch_size = x.size(0)

        # Backbone
        resnet_feat, layer1_feat, layer2_feat, layer3_feat = self.backbone(x)  # (batch_size, 2048, 16, 8)
        hierarchical_feats = [layer1_feat, layer2_feat, layer3_feat]

        # Gloab module ([N, 2048])
        gloab_feat = self.pool_layer(resnet_feat)  # (batch_size, 2048, 1, 1)
        gloab_feat = gloab_feat.view(batch_size, -1)  # (batch_size, 2048)
        norm_gloab_feat = self.gloab_bottleneck(gloab_feat)  # (batch_size, 2048)

        hierarchical_feat_list = [self.hierarchical_PoolBn_list[i](hierarchical_feats[i].view(batch_size, -1)) for i in range(self.hierarchical_parts)]

        if self.training:
            # Gloab module to classifier([N, num_classes]）
            gloab_score = self.gloab_classifier(norm_gloab_feat)

            hierarchical_score_list = [self.hierarchical_FC_list[i](hierarchical_feat_list[i].view(batch_size, -1)) for i in range(self.hierarchical_parts)]

            return (
                gloab_score,
                gloab_feat,
                hierarchical_score_list,
                hierarchical_feat_list,
            )

        else:
            return norm_gloab_feat
