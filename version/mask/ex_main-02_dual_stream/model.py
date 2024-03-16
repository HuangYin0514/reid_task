import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class Resnet50_Baseline(nn.Module):
    def __init__(self):
        super(Resnet50_Baseline, self).__init__()

        # Backbone
        resnet = network.backbones.resnet50(pretrained=True)

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
    def __init__(self, num_classes, **kwargs):

        super(ReidNet, self).__init__()
        self.num_classes = num_classes

        # Backbone
        self.backbone = Resnet50_Baseline()

        # Gloab module
        self.gloab_avgpool = nn.AdaptiveAvgPool2d(1)
        self.gloab_bottleneck = nn.BatchNorm1d(2048)
        self.gloab_bottleneck.bias.requires_grad_(False)
        self.gloab_classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.gloab_bottleneck.apply(network.utils.weights_init_kaiming)
        self.gloab_classifier.apply(network.utils.weights_init_classifier)

        # Gloab module
        self.mask_avgpool = nn.AdaptiveAvgPool2d(1)
        self.mask_bottleneck = nn.BatchNorm1d(2048)
        self.mask_bottleneck.bias.requires_grad_(False)
        self.mask_classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.mask_bottleneck.apply(network.utils.weights_init_kaiming)
        self.mask_classifier.apply(network.utils.weights_init_classifier)

    def heatmap(self, x):
        return self.backbone(x)

    def forward(self, x, mask):
        batch_size = x.size(0)

        resnet_feat = self.backbone(x)
        mask_feat = self.backbone(mask)

        # Gloab module ([N, 2048])
        gloab_feat = self.gloab_avgpool(resnet_feat)  # (batch_size, 2048, 1, 1)
        gloab_feat = gloab_feat.view(batch_size, -1)  # (batch_size, 2048)
        norm_gloab_feat = self.gloab_bottleneck(gloab_feat)  # (batch_size, 2048)

        # Mask module ([N, 2048])
        mask_feat = self.mask_avgpool(resnet_feat)  # (batch_size, 2048, 1, 1)
        mask_feat = mask_feat.view(batch_size, -1)  # (batch_size, 2048)
        norm_mask_feat = self.mask_bottleneck(mask_feat)  # (batch_size, 2048)

        if self.training:
            # Gloab module to classifier([N, num_classes]）
            gloab_score = self.gloab_classifier(norm_gloab_feat)

            # Mask module to classifier([N, num_classes]）
            mask_score = self.mask_classifier(norm_mask_feat)
            return gloab_score, gloab_feat, mask_score, mask_feat

        else:
            return norm_gloab_feat
