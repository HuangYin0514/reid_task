import torch
import torch.nn.functional as F
from torch import nn


class FeatureMapLocalizedIntegratingNoRelu:
    def __init__(self, config):
        super(FeatureMapLocalizedIntegratingNoRelu, self).__init__()
        self.config = config

    def __call__(self, features_map, pids, classifier):
        bs = features_map.size(0)
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)

        # 提取 classifier 参数, classifier_params [name, param]
        classifier_params = list(classifier.named_parameters())[-1]
        params_selected = classifier_params[1]

        # 获得热图
        heatmaps_raw = torch.einsum("bc, bcij -> bij", params_selected[pids], features_map.detach())

        # 归一化热图
        heatmaps_min = heatmaps_raw.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        heatmaps_max = heatmaps_raw.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        heatmap_range = heatmaps_max - heatmaps_min
        heatmap_range[heatmap_range == 0] = 1
        heatmaps_normalized = (heatmaps_raw - heatmaps_min) / heatmap_range  # torch.Size([60, 24, 8])

        # 获得关注区域
        localized_features_map = features_map * heatmaps_normalized.unsqueeze(1)

        return localized_features_map  # torch.Size([60, 2048, 24, 8])


class FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights:
    def __init__(self, config):
        super(FeatureMapQuantifiedIntegratingProbLogSoftmaxWeights, self).__init__()
        self.config = config

    def __call__(self, features_map, cls_scores, pids):
        bs = features_map.size(0)
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)

        prob = torch.log_softmax(cls_scores, dim=1)
        probs = prob[torch.arange(bs), pids]

        chunk_size = int(bs / 4)  # 15
        chunk_probs = torch.chunk(probs, chunks=chunk_size, dim=0)
        chunk_weights = [torch.softmax(chunk, dim=0) for chunk in chunk_probs]
        weights = torch.cat(chunk_weights, dim=0)  # torch.Size([60])
        quantified_features_map = weights.view(-1, 1, 1, 1) * features_map  # torch.Size([60, 2048, 24, 8])

        chunk_quantified_features_map = torch.chunk(quantified_features_map, chunks=chunk_size, dim=0)
        chunk_pids = torch.chunk(pids, chunks=chunk_size, dim=0)

        chunk_quantified_features_map = torch.stack(chunk_quantified_features_map, dim=0)  # torch.Size([15, 4, 2048, 24, 8])
        quantified_integrating_features_map = chunk_quantified_features_map.sum(dim=1)  # torch.Size([15, 2048, 24, 8])

        integrating_pids = torch.stack([chunk[0] for chunk in chunk_pids], dim=0)  # torch.Size([15])

        return quantified_features_map, quantified_integrating_features_map, integrating_pids


class ReasoningLoss(nn.Module):
    def __init__(self):
        super(ReasoningLoss, self).__init__()

    def __call__(self, bn_features, bn_features2):
        new_bn_features2 = torch.zeros(bn_features.size()).cuda()
        for i in range(int(bn_features2.size(0) / 4)):
            new_bn_features2[i * 4 : i * 4 + 4] = bn_features2[i]
        loss = torch.norm((bn_features - new_bn_features2), p=2)
        return loss
