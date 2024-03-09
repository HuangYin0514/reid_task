import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class SAMS(nn.Module):
    """
    Split-Attend-Merge-Stack agent
    Input an feature map with shape H*W*C, we first split the feature maps into
    multiple parts, obtain the attention map of each part, and the attention map
    for the current pyramid level is constructed by mergiing each attention map.
    """

    def __init__(
        self,
        in_channels,
        channels,
        radix=4,
        reduction_factor=4,
        norm_layer=nn.BatchNorm2d,
    ):
        super(SAMS, self).__init__()
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.channels = channels
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=1)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=1)

    def forward(self, x):

        batch, channel = x.shape[:2]
        splited = torch.split(x, channel // self.radix, dim=1)

        gap = sum(splited)
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap).view((batch, self.radix, self.channels))
        atten = F.softmax(atten, dim=1).view(batch, -1, 1, 1)
        atten = torch.split(atten, channel // self.radix, dim=1)

        out = torch.cat([att * split for (att, split) in zip(atten, splited)], 1)
        return out.contiguous()


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class BN2d(nn.Module):
    def __init__(self, planes):
        super(BN2d, self).__init__()
        self.bottleneck2 = nn.BatchNorm2d(planes)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck2.apply(network.utils.weights_init_kaiming)

    def forward(self, x):
        return self.bottleneck2(x)


class Resnet50_Branch(nn.Module):
    def __init__(self):
        super(Resnet50_Branch, self).__init__()

        # Backbone
        # resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
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

        # New layers
        self.level = 2
        self.att2 = SELayer(256, 32)
        self.att3 = SELayer(512, 64)
        self.att4 = SELayer(1024, 128)
        self.att5 = SELayer(2048, 256)

        self.att_s2 = SAMS(256, int(256 / self.level), radix=self.level)
        self.att_s3 = SAMS(512, int(512 / self.level), radix=self.level)
        self.att_s4 = SAMS(1024, int(1024 / self.level), radix=self.level)
        self.att_s5 = SAMS(2048, int(2048 / self.level), radix=self.level)

        self.BN2 = BN2d(256)
        self.BN3 = BN2d(512)
        self.BN4 = BN2d(1024)
        self.BN5 = BN2d(2048)

        self.att_ss2 = SAMS(256, int(256 / self.level), radix=self.level)
        self.att_ss3 = SAMS(512, int(512 / self.level), radix=self.level)
        self.att_ss4 = SAMS(1024, int(1024 / self.level), radix=self.level)
        self.att_ss5 = SAMS(2048, int(2048 / self.level), radix=self.level)

        self.BN_2 = BN2d(256)
        self.BN_3 = BN2d(512)
        self.BN_4 = BN2d(1024)
        self.BN_5 = BN2d(2048)

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        x = self.resnet_layer1(x)
        x = self.att_ss2(x)
        x = self.BN_2(x)
        x = self.att_s2(x)
        x = self.BN2(x)
        y = self.att2(x)
        x = x * y.expand_as(x)

        x = self.resnet_layer2(x)
        x = self.att_ss3(x)
        x = self.BN_3(x)
        x = self.att_s3(x)
        x = self.BN3(x)
        y = self.att3(x)
        x = x * y.expand_as(x)

        x = self.resnet_layer3(x)
        x = self.att_ss4(x)
        x = self.BN_4(x)
        x = self.att_s4(x)
        x = self.BN4(x)
        y = self.att4(x)
        x = x * y.expand_as(x)

        x = self.resnet_layer4(x)
        x = self.att_ss5(x)
        x = self.BN_5(x)
        x = self.att_s5(x)
        x = self.BN5(x)
        y = self.att5(x)
        x = x * y.expand_as(x)

        return x


class Feature_Fusion_Module(nn.Module):
    def __init__(self, parts, **kwargs):
        super(Feature_Fusion_Module, self).__init__()

        self.parts = parts

        self.fc1 = nn.Linear(256, 6)
        self.fc1.apply(network.utils.weights_init_kaiming)

    def forward(self, gloab_feature, parts_features):
        batch_size = gloab_feature.size(0)

        # Compute the weigth of parts features
        w_of_parts = torch.sigmoid(self.fc1(gloab_feature))

        # Compute the features,with weigth
        weighted_feature = torch.zeros_like(parts_features[0])
        for i in range(self.parts):
            new_feature = parts_features[i] * w_of_parts[:, i].view(batch_size, 1, 1).expand(parts_features[i].shape)
            weighted_feature += new_feature

        return weighted_feature.squeeze()


class ReidNet(nn.Module):
    def __init__(self, num_classes, **kwargs):

        super(ReidNet, self).__init__()
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
        self.part_classifier_list = nn.ModuleList()
        for _ in range(self.parts):
            fc = nn.Linear(256, num_classes)
            nn.init.normal_(fc.weight, std=0.001)
            nn.init.constant_(fc.bias, 0)
            self.part_classifier_list.append(fc)

        # Gloab module
        ## Network
        self.k11_conv = nn.Conv2d(2048, 512, kernel_size=1)
        self.gloab_agp = nn.AdaptiveAvgPool2d((1, 1))
        self.gloab_conv = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
        self.gloab_conv.apply(network.utils.weights_init_kaiming)
        # ## Classifier
        # self.gloab_classifier = nn.Linear(256, num_classes)
        # nn.init.normal_(self.gloab_classifier.weight, std=0.001)
        # nn.init.constant_(self.gloab_classifier.bias, 0)

        # Feature fusion module
        self.ffm = Feature_Fusion_Module(self.parts)
        self.fusion_feature_classifier = nn.Linear(256, num_classes)
        nn.init.normal_(self.fusion_feature_classifier.weight, std=0.001)
        nn.init.constant_(self.fusion_feature_classifier.bias, 0)

    def heatmap(self, x):
        return self.backbone(x)

    def forward(self, x):
        batch_size = x.size(0)

        # Backbone ([N, 2048, 24, 8])
        resnet_feat = self.backbone(x)

        # Part module (6 x [N, 256, 1])
        feat_G = self.avgpool(resnet_feat)
        part_feat = []
        for i in range(self.parts):
            stripe_feat_H = self.local_conv_list[i](feat_G[:, :, i, :])
            part_feat.append(stripe_feat_H)

        # Gloab module ([N, 256])
        gloab_feat = self.k11_conv(resnet_feat)
        gloab_feat = self.gloab_agp(gloab_feat).view(batch_size, 512, -1)
        gloab_feat = self.gloab_conv(gloab_feat).squeeze()

        # Feature fusion module ([N, 256])
        fusion_feat = self.ffm(gloab_feat, part_feat)

        if self.training:
            # Classifier for parts module (6 x [N, num_classes]）
            part_score_list = [self.part_classifier_list[i](part_feat[i].view(batch_size, -1)) for i in range(self.parts)]

            part_feat = torch.cat(part_feat, dim=2)
            part_feat = F.normalize(part_feat, p=2, dim=1)
            part_feat = part_feat.view(batch_size, -1)
            return part_score_list, part_feat, gloab_feat, fusion_feat
        else:
            # Part features ([N, 1536])
            part_feat = torch.cat(part_feat, dim=2)
            part_feat = F.normalize(part_feat, p=2, dim=1)
            part_feat = part_feat.view(batch_size, -1)

            # # Gloab features ([N, 256])
            # gloab_feat = F.normalize(gloab_feat, p=2, dim=1)
            # gloab_feat = gloab_feat.view(batch_size, -1)

            # # Gloab features ([N, 1792])
            # all_feat = torch.cat([part_feat, gloab_feat], dim=-1)

            return part_feat
