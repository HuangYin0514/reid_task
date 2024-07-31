from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision import models

import network


class Scaled_dot_product_attention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values

    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, dim: int):
        super(Scaled_dot_product_attention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float("Inf"))

        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context, attn


class Multi_head_attention(nn.Module):
    """
    Multi-Head Attention proposed in "Attention Is All You Need"
    Instead of performing a single attention function with d_model-dimensional keys, values, and queries,
    project the queries, keys and values h times with different, learned linear projections to d_head dimensions.
    These are concatenated and once again projected, resulting in the final values.
    Multi-head attention allows the model to jointly attend to information from different representation
    subspaces at different positions.

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_o
        where head_i = Attention(Q · W_q, K · W_k, V · W_v)

    Args:
        d_model (int): The dimension of keys / values / quries (default: 512)
        num_heads (int): The number of attention heads. (default: 8)

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): In transformer, three different ways:
            Case 1: come from previoys decoder layer
            Case 2: come from the input embedding
            Case 3: come from the output embedding (masked)

        - **key** (batch, k_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **value** (batch, v_len, d_model): In transformer, three different ways:
            Case 1: come from the output of the encoder
            Case 2: come from the input embeddings
            Case 3: come from the output embedding (masked)

        - **mask** (-): tensor containing indices to be masked

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features.
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoder outputs.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(Multi_head_attention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.scaled_dot_attn = Scaled_dot_product_attention(self.d_head)
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)  # BxQ_LENxNxD
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head)  # BxK_LENxNxD
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head)  # BxV_LENxNxD

        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxQ_LENxD
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxK_LENxD
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.num_heads, -1, self.d_head)  # BNxV_LENxD

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # BxNxQ_LENxK_LEN

        context, attn = self.scaled_dot_attn(query, key, value, mask)

        context = context.view(self.num_heads, batch_size, -1, self.d_head)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.num_heads * self.d_head)  # BxTxND

        return context, attn


class Channel_attention(nn.Module):
    # https://github.com/EvilPsyCHo/Attention-PyTorch
    def __init__(self, in_planes, rotio=16):
        super(Channel_attention, self).__init__()

        rotio = 4  # Mine

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(nn.Conv2d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(), nn.Conv2d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class Spatial_attention(nn.Module):
    # https://github.com/EvilPsyCHo/Attention-PyTorch
    def __init__(self, kernel_size=7):
        super(Spatial_attention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class Integrate_feats_module(nn.Module):
    def __init__(self, classifier_head, config, logger):
        super(Integrate_feats_module, self).__init__()
        self.config = config
        self.logger = logger

        # CAM classifier head
        self.classifier_head = classifier_head

        # CBAM attention
        self.ca = Channel_attention(in_planes=4)  # planes是feature map的通道个数
        self.sa = Spatial_attention()
        self.ca.apply(network.utils.weights_init_kaiming)
        self.sa.apply(network.utils.weights_init_kaiming)

        self.multi_head_attention = Multi_head_attention(d_model=128, num_heads=8)

        # Integrate_layer
        conv11 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1, bias=False)
        bn = nn.BatchNorm2d(1)
        act = nn.ReLU(inplace=True)
        self.integrate_layer = nn.Sequential(conv11)
        self.integrate_layer.apply(network.utils.weights_init_kaiming)

    def forward(self, feats, pids, num_same_id=4):
        bs = feats.size(0)
        c, h, w = feats.size(1), feats.size(2), feats.size(3)
        chunk_size = int(bs / num_same_id)  # 15

        # CAM
        CAM_feats = self._cam(feats, pids)  #  (bs, 1, h, w)
        CAM_feats_reshaped = CAM_feats.view(chunk_size, num_same_id, h, w)  # (chunk_size, 4, h, w)
        # CAM_feats_reshaped = F.relu(CAM_feats_reshaped)  # (chunk_size, 4, h, w)

        # Attention
        # CBAM_feats = self.ca(CAM_feats_reshaped) * CAM_feats_reshaped  # (bs, 4, h, w)
        # CBAM_feats = 0.25 * CAM_feats_reshaped  # (bs, 4, h, w)
        Multi_head_attention_input = CAM_feats_reshaped.view(chunk_size, num_same_id, h * w)  # (bs, 4, h*w)
        attention_feats, attention_matrix = self.multi_head_attention(Multi_head_attention_input, Multi_head_attention_input, Multi_head_attention_input)  # (bs, 4, h*w)
        attention_feats = attention_feats.view(chunk_size, num_same_id, h, w)  # (bs, 4, h, w)

        print("multi_head_attention_input.shape: ", Multi_head_attention_input.shape)
        print("attention_feats.shape: ", attention_feats.shape)

        # Integrate
        integrate_feats = torch.sum(attention_feats, dim=1, keepdim=True)  # (chunk_size, 1, h, w)
        integrate_pids = pids[::num_same_id]  # 直接从 pids 中获取 integrate_pids

        return integrate_feats, integrate_pids

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
        backbone_pool_feats, backbone_bn_feats, backbone_cls_score = self.classifier_head(resnet_feats)

        if self.training:
            return backbone_cls_score, backbone_pool_feats, backbone_bn_feats, resnet_feats
        else:
            return backbone_bn_feats
