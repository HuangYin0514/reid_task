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


class Bottleneck(nn.Module):
    def __init__(self, input_dim):
        super(Bottleneck, self).__init__()

        self.bottleneck = nn.BatchNorm1d(input_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(network.utils.weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck(x)


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()

        self.classifier = nn.Linear(input_dim, output_dim, bias=False)
        self.classifier.apply(network.utils.weights_init_classifier)

    def forward(self, x):
        return self.classifier(x)


class ReidNet(nn.Module):
    def __init__(self, num_classes, **kwargs):

        super(ReidNet, self).__init__()
        self.num_classes = num_classes

        # Backbone
        self.backbone = Resnet50_Baseline()

        # Gloab module
        self.gloab_avgpool = nn.AdaptiveAvgPool2d(1)
        self.gloab_bottleneck = Bottleneck(2048)
        self.gloab_classifier = Classifier(2048, self.num_classes)

    def heatmap(self, x):
        return self.backbone(x)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        resnet_feat = self.backbone(x)  # (bs, 2048, 16, 8)

        # Gloab module
        gloab_feat = self.gloab_avgpool(resnet_feat)  # (bs, 2048, 1, 1)
        gloab_feat = gloab_feat.view(bs, -1)  # (bs, 2048)
        norm_gloab_feat = self.gloab_bottleneck(gloab_feat)  # (bs, 2048)

        if self.training:
            # Gloab module to classifier
            gloab_score = self.gloab_classifier(norm_gloab_feat)  # ([bs, num_classes]）
            return gloab_score, gloab_feat

        else:
            return norm_gloab_feat
