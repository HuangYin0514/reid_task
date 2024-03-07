import torch
import torch.nn as nn
import torch.nn.functional as F

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


# apnet修改的模块
class Resnet_Backbone(nn.Module):
    def __init__(self):
        super(Resnet_Backbone, self).__init__()

        # backbone--------------------------------------------------------------------------
        # change the model different from pcb
        resnet = network.backbones.resnet50(pretrained=True)
        # Modifiy the stride of last conv layer----------------------------
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        # Remove avgpool and fc layer of resnet------------------------------
        self.resnet_conv1 = resnet.conv1
        self.resnet_bn1 = resnet.bn1
        self.resnet_relu = resnet.relu
        self.resnet_maxpool = resnet.maxpool
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

        # apnet模块
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


class baseline_apnet(nn.Module):
    def __init__(self, num_classes):

        self.num_classes = num_classes

        super(baseline_apnet, self).__init__()

        # backbone
        self.backbone = Resnet_Backbone()

        # baseline
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.bottleneck.apply(network.utils.weights_init_kaiming)
        self.classifier.apply(network.utils.weights_init_classifier)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.backbone(x)  # (batch_size, 2048, 16, 8)

        # baseline
        x = self.avgpool(x)  # (batch_size, 2048, 1, 1)
        x = x.view(batch_size, -1)  # (batch_size, 2048)
        feat = self.bottleneck(x)  # (batch_size, 2048)

        if self.training:
            score = self.classifier(feat)  # (batch_size, num_classes)
            return score, x
        else:
            return feat
