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
        heatmaps_min = heatmaps_raw.amin(dim=(-2, -1), keepdim=True)
        heatmaps_max = heatmaps_raw.amax(dim=(-2, -1), keepdim=True)
        heatmap_range = heatmaps_max - heatmaps_min
        heatmap_range[heatmap_range == 0] = 1
        heatmaps_normalized = (heatmaps_raw - heatmaps_min) / heatmap_range  # torch.Size([60, 24, 8])

        # 获得关注区域
        localized_features_map = features_map * heatmaps_normalized.unsqueeze(1)

        return localized_features_map  # torch.Size([60, 2048, 24, 8])


class ReasoningLoss(nn.Module):
    def __init__(self):
        super(ReasoningLoss, self).__init__()

    def __call__(self, feat, feat2):
        new_feat = feat2.repeat_interleave(4, dim=0)
        loss = torch.norm(feat - new_feat, p=2)
        return loss
