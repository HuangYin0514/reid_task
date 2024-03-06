import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class Resnet50_Branch(nn.Module):
    def __init__(self, **kwargs):
        super(Resnet50_Branch, self).__init__()

        # Backbone
        # resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        resnet = network.backbones.resnet50(pretrained=True)

        # Modifiy backbone
        ## Modifiy the stride of last conv layer
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        ## Remove avgpool and fc layer of resnet
        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)

    def forward(self, x):
        return self.backbone(x)


class PCBModel(nn.Module):
    def __init__(self, num_classes, **kwargs):

        super(PCBModel, self).__init__()
        self.parts = 6
        self.num_classes = num_classes

        # Backbone
        self.backbone = Resnet50_Branch()

        # Part module
        ## Avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        # self.dropout = nn.Dropout(p=0.5)

        ## Local_conv
        self.local_conv_list = nn.ModuleList()
        for _ in range(self.parts):
            local_conv = nn.Sequential(nn.Conv1d(2048, 256, kernel_size=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
            self.local_conv_list.append(local_conv)

        ## Classifier
        self.fc_list = nn.ModuleList()
        for _ in range(self.parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.fc_list.append(fc)

    def forward(self, x):
        # Backbone ([N, 2048, 24, 8])
        resnet_features = self.backbone(x)

        # Avgpool ([N, 2048, 6, 1])
        features_G = self.avgpool(resnet_features)

        # Local_conv (6 x [N, 256, 1])
        features_H = []
        for i in range(self.parts):
            stripe_features_H = self.local_conv_list[i](features_G[:, :, i, :])
            features_H.append(stripe_features_H)

        if self.training:
            # Classifier for parts module ([N, num_classes]）
            batch_size = x.size(0)
            logits_list = [self.fc_list[i](features_H[i].view(batch_size, -1)) for i in range(self.parts)]
            return logits_list
        else:
            # Cat features ([N, 1536+512])
            v_g = torch.cat(features_H, dim=2)
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)


def PCB(num_classes, **kwargs):
    return PCBModel(num_classes=num_classes, **kwargs)
