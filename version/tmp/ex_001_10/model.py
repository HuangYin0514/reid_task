import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import functional as F
from torchvision import models

import network


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, config, logger):
        super(GCNLayer, self).__init__()
        self.config = config
        self.logger = logger

        self.adj = 0.25 * torch.ones(4, 4, device=self.config.device)
        self.W = nn.Parameter(torch.rand(input_dim, output_dim))

    def forward(self, feats):
        """L * X * W"""
        out = torch.einsum("bcx,xy->bcy", feats, self.W)
        out = torch.einsum("yc,bcx->byx", self.adj, out)
        return out


class Integrate_feats_module(nn.Module):
    def __init__(self, classifier_head, config, logger):
        super(Integrate_feats_module, self).__init__()
        self.config = config
        self.logger = logger

        self.classifier_head = classifier_head

        self.gcn_l1 = GCNLayer(128, 128, config, logger)
        self.gcn_l2 = GCNLayer(128, 128, config, logger)
        self.gcn_w3 = nn.Parameter(torch.rand(4, 1))

    def forward(self, feats, pids, num_same_id=4):
        bs = feats.size(0)
        c, h, w = feats.size(1), feats.size(2), feats.size(3)
        chunk_size = int(bs / num_same_id)  # 15

        # CAM
        weighted_feats = self._cam(feats, pids)  #  (bs, 1, h, w)
        weighted_feats_reshaped = weighted_feats.view(chunk_size, num_same_id, h, w)  # (chunk_size, 4, h, w)

        # GCN
        integrate_feats = self._gcn(weighted_feats_reshaped)  # (chunk_size, 1, h, w)
        integrate_pids = pids[::num_same_id]  # 直接从 pids 中获取 integrate_pids

        return integrate_feats, integrate_pids

    def _gcn(self, feats):
        bs = feats.size(0)
        c, h, w = feats.size(1), feats.size(2), feats.size(3)

        feats = feats.view(bs, c, h * w)  # (15, 4, 128)
        out = self.gcn_l1(feats)  # torch.Size([15, 4, 128])
        out = F.elu(out)
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.gcn_l2(out)  # torch.Size([15, 4, 128])
        out = torch.sum(out, dim=1)  # torch.Size([15, 128])
        out = F.normalize(out, dim=1)
        out = out.view(bs, 1, h, w)  # torch.Size([15, 1, 16, 8])
        return out

    def _cam(self, feats, pids):
        # Classifier parameters
        classifier_name = []
        classifier_params = []
        for name, param in self.classifier_head.classifier.named_parameters():
            classifier_name.append(name)
            classifier_params.append(param)
        # cam
        weighted_feats = torch.einsum("bc,bchw->bhw", classifier_params[-1][pids], feats).unsqueeze(1)
        return weighted_feats


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
        pool_feat = None  # torch.Size([batch_size, 1, 16, 8])
        pool_feat = feat.view(bs, -1)  # (batch_size, 128)
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
        self.auxiliary_classifier_head = Auxiliary_classifier_head(128, num_classes, config, logger)

        # Integrat Feats Module
        self.integrate_feats_module = Integrate_feats_module(self.classifier_head, config, logger)

    def forward(self, x):
        bs = x.size(0)

        # Backbone
        resnet_feats = self.backbone(x)  # (bs, 2048, 16, 8)

        # Classifier head
        g_pool_feats, g_bn_feats, g_cls_score = self.classifier_head(resnet_feats)

        if self.training:
            return g_cls_score, g_pool_feats, g_bn_feats, resnet_feats
        else:
            return g_bn_feats
