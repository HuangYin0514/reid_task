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


class Feature_Fusion_Module(nn.Module):
    def __init__(self, **kwargs):
        super(Feature_Fusion_Module, self).__init__()
        self.parts = 6

        # Part module
        self.part_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.part_local_conv_list = nn.ModuleList()
        for _ in range(self.parts):
            local_conv = nn.Sequential(nn.Conv1d(2048, 256, kernel_size=1), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
            self.part_local_conv_list.append(local_conv)

        self.fc = nn.Linear(3072, 1536, bias=False)
        self.fc.apply(network.utils.weights_init_classifier)

    def forward(self, feat_1, feat_2):
        bs = feat_1.size(0)
        feat_1 = self.part_avgpool(feat_1)
        f1_feats = []
        for i in range(self.parts):
            stripe_feat_H = self.part_local_conv_list[i](feat_1[:, :, i, :])
            f1_feats.append(stripe_feat_H)

        feat_2 = self.part_avgpool(feat_2)
        f2_feats = []
        for i in range(self.parts):
            stripe_feat_H = self.part_local_conv_list[i](feat_2[:, :, i, :])
            f2_feats.append(stripe_feat_H)

        f1_cat = torch.cat(f1_feats, dim=-1).reshape(bs, -1)
        f2_cat = torch.cat(f2_feats, dim=-1).reshape(bs, -1)

        f_fusion = torch.cat([f1_cat, f2_cat], dim=-1)
        f_fusion = self.fc(f_fusion)
        return f_fusion


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

        # Mask module
        self.mask_avgpool = nn.AdaptiveAvgPool2d(1)
        self.mask_bottleneck = nn.BatchNorm1d(2048)
        self.mask_bottleneck.bias.requires_grad_(False)
        self.mask_bottleneck.apply(network.utils.weights_init_kaiming)
        self.mask_classifier = nn.Linear(2048, self.num_classes, bias=False)
        self.mask_classifier.apply(network.utils.weights_init_classifier)

        # Fusion module
        self.fusion_module = Feature_Fusion_Module()
        self.fusion_bottleneck = nn.BatchNorm1d(1536)
        self.fusion_bottleneck.bias.requires_grad_(False)
        self.fusion_bottleneck.apply(network.utils.weights_init_kaiming)
        self.fusion_classifier = nn.Linear(1536, self.num_classes, bias=False)
        self.fusion_classifier.apply(network.utils.weights_init_classifier)

    def heatmap(self, x):
        return self.backbone(x)

    def forward(self, x, mask):
        batch_size = x.size(0)

        resnet_feat = self.backbone(x)
        mask_backbone_feat = self.backbone(mask)

        # Gloab module ([N, 2048])
        gloab_feat = self.gloab_avgpool(resnet_feat)  # (batch_size, 2048, 1, 1)
        gloab_feat = gloab_feat.view(batch_size, -1)  # (batch_size, 2048)
        norm_gloab_feat = self.gloab_bottleneck(gloab_feat)  # (batch_size, 2048)

        # Mask module ([N, 2048])
        mask_feat = self.mask_avgpool(mask_backbone_feat)  # (batch_size, 2048, 1, 1)
        mask_feat = mask_feat.view(batch_size, -1)  # (batch_size, 2048)
        norm_mask_feat = self.mask_bottleneck(mask_feat)  # (batch_size, 2048)

        # Fusion module ([N, 1536])
        fusion_feat = self.fusion_module(resnet_feat, mask_backbone_feat)
        fusion_feat = fusion_feat.view(batch_size, -1)  # (batch_size, 2048)
        norm_fusion_feat = self.fusion_bottleneck(fusion_feat)  # (batch_size, 2048)

        if self.training:
            # Gloab module to classifier([N, num_classes]）
            gloab_score = self.gloab_classifier(norm_gloab_feat)

            # Mask module to classifier([N, num_classes]）
            mask_score = self.mask_classifier(norm_mask_feat)

            # Fusion module to classifier([N, num_classes]）
            fusion_score = self.fusion_classifier(norm_fusion_feat)
            return gloab_score, gloab_feat, mask_score, mask_feat, fusion_score, fusion_feat

        else:
            return norm_gloab_feat
