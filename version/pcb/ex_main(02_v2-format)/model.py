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
        self.avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))  ## Avgpool
        self.local_conv_list = nn.ModuleList()  ## Local_conv
        for _ in range(self.parts):
            local_conv = nn.Sequential(nn.Conv1d(2048, 256, kernel_size=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
            self.local_conv_list.append(local_conv)
        self.part_classifier_list = nn.ModuleList()  ## Classifier
        for _ in range(self.parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.part_classifier_list.append(fc)

    def forward(self, x):
        batch_size = x.size(0)

        # Backbone ([N, 2048, 24, 8])
        resnet_feat = self.backbone(x)

        # Part module
        feat_G = self.avgpool(resnet_feat)  ## Avgpool ([N, 2048, 6, 1])
        part_feat_list = []  ## Local_conv (6 x [N, 256, 1])
        for i in range(self.parts):
            stripe_feat = self.local_conv_list[i](feat_G[:, :, i, :])
            part_feat_list.append(stripe_feat)

        if self.training:
            # Classifier for part module (6 x [N, num_classes]）
            part_score = [self.part_classifier_list[i](part_feat_list[i].view(batch_size, -1)) for i in range(self.parts)]
            return part_score
        else:
            # Part features ([N, 1536])
            part_feat = torch.cat(part_feat_list, dim=2)
            part_feat = F.normalize(part_feat, p=2, dim=1)
            part_feat = part_feat.view(part_feat.size(0), -1)
            return part_feat


def PCB(num_classes, **kwargs):
    return PCBModel(num_classes=num_classes, **kwargs)
