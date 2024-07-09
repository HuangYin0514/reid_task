import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class IntegratFeatureMap(nn.Module):
    def __init__(self, config, logger):
        super(IntegratFeatureMap, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, feats_map, pids, classifier):
        bs = feats_map.size(0)
        c, h, w = feats_map.size(1), feats_map.size(2), feats_map.size(3)
        chunk_size = int(bs / 4)  # 15

        classifier_params = list(classifier.named_parameters())[-1]
        params_selected = classifier_params[1]
        feats_map_raw = torch.einsum("bc, bcij -> bij", params_selected[pids], feats_map.detach()).detach()

        feats_map_raw = feats_map_raw.view(bs, h * w)
        feats_map_raw = F.normalize(feats_map_raw, p=2, dim=1)
        feats_map_raw = feats_map_raw.view(bs, h, w)

        localized_features_map = feats_map * feats_map_raw.unsqueeze(1)

        feats_map_reshaped = localized_features_map.view(chunk_size, 4, c, h, w)  # 将 feats_map 重塑为 (chunk_size, 4, 1, h, w)
        integrat_feats_map = 0.25 * torch.sum(feats_map_reshaped, dim=1).cuda()  # 计算 integrat_feats_map，shape 变为 (chunk_size, c, h, w)
        integrat_pids = pids[::4]  # 直接从 pids 中获取 integrat_pids

        return integrat_feats_map, integrat_pids


class Auxiliary_classifier_head(nn.Module):
    def __init__(self, feat_dim, num_classes, config, logger, **kwargs):
        super(
            Auxiliary_classifier_head,
            self,
        ).__init__()
        self.config = config
        self.logger = logger

        # Pooling
        self.pool_layer = nn.AdaptiveAvgPool2d(1)

        # BatchNorm
        self.BN = nn.BatchNorm1d(feat_dim)
        self.BN.bias.requires_grad_(False)
        self.BN.apply(network.utils.weights_init_kaiming)

        # Classifier
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.classifier.apply(network.utils.weights_init_classifier)

    def forward(self, feat):  # (batch_size, dim)
        bs = feat.size(0)
        # pool
        pool_feat = self.pool_layer(feat)  # (batch_size, 2048, 1, 1)
        pool_feat = pool_feat.view(bs, -1)  # (batch_size, 2048)
        # BN
        bn_feat = self.BN(pool_feat)  # (batch_size, 2048)
        # Classifier
        cls_score = self.classifier(bn_feat)  # ([N, num_classes]）
        return pool_feat, bn_feat, cls_score


class Classifier_head(nn.Module):
    def __init__(self, feat_dim, num_classes, config, logger, **kwargs):
        super(
            Classifier_head,
            self,
        ).__init__()
        self.config = config
        self.logger = logger

        # Pooling
        self.pool_layer = nn.AdaptiveAvgPool2d(1)

        # BatchNorm
        self.BN = nn.BatchNorm1d(feat_dim)
        self.BN.bias.requires_grad_(False)
        self.BN.apply(network.utils.weights_init_kaiming)

        # Classifier
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        self.classifier.apply(network.utils.weights_init_classifier)

    def forward(self, feat):  # (batch_size, dim)
        bs = feat.size(0)
        # pool
        pool_feat = self.pool_layer(feat)  # (batch_size, 2048, 1, 1)
        pool_feat = pool_feat.view(bs, -1)  # (batch_size, 2048)
        # BN
        bn_feat = self.BN(pool_feat)  # (batch_size, 2048)
        # Classifier
        cls_score = self.classifier(bn_feat)  # ([N, num_classes]）
        return pool_feat, bn_feat, cls_score


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        # Backbone
        resnet = network.backbones.resnet50(pretrained=True)
        # resnet = network.backbones.resnet50_ibn_a(pretrained=True)

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

        # Classifier head
        self.classifier_head = Classifier_head(2048, num_classes, config, logger)

        # Auxiliary classifier
        self.auxiliary_classifier_head = Auxiliary_classifier_head(2048, num_classes, config, logger)

        # IntegratFeatureMap
        self.integratFeatureMap = IntegratFeatureMap(config, logger)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        resnet_feat = self.backbone(x)  # (bs, 2048, 16, 8)

        # Classifier head
        G_pool_feat, G_bn_feat, G_cls_score = self.classifier_head(resnet_feat)

        if self.training:
            return G_cls_score, G_pool_feat, G_bn_feat, resnet_feat
        else:
            return G_bn_feat
