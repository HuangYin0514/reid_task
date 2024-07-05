
import torch
import torch.nn.functional as F
from torch import nn


class IntegratWeightedFeatureMap:
    def __init__(self, config):
        super(IntegratWeightedFeatureMap, self).__init__()
        self.config = config

    def __call__(self, features_map, cls_scores, pids):
        bs = features_map.size(0)
        c, h, w = features_map.size(1), features_map.size(2), features_map.size(3)

        chunk_size = int(bs / 4)  # 15

        probs = cls_scores[torch.arange(bs), pids]
        chunk_probs = torch.chunk(probs, chunks=chunk_size, dim=0)
        chunk_pids = torch.chunk(pids, chunks=chunk_size, dim=0)

        # Weights
        weights = torch.zeros(bs, 1).to(self.config.device)
        for idx, chunk in enumerate(chunk_probs):
            weights[4 * idx : 4 * (idx + 1)] = torch.softmax(chunk, dim=0).reshape(-1, 1)

        # weighted features map
        weighted_features_map = torch.zeros(features_map.size()).cuda()
        for i in range(bs):
            weighted_features_map[i] = 0.25 * features_map[i]

        integrat_weighted_features_map = torch.zeros([chunk_size, c, h, w]).cuda()
        integrating_pids = torch.zeros([chunk_size]).to(torch.int64)
        for i in range(chunk_size):
            integrat_weighted_features_map[i, :, :, :] = (
                weighted_features_map[4 * i] + weighted_features_map[4 * i + 1] + weighted_features_map[4 * i + 2] + weighted_features_map[4 * i + 3]
            )
            integrating_pids[i] = chunk_pids[i][0]

        return integrat_weighted_features_map, integrating_pids


class ReasoningLoss(nn.Module):
    def __init__(self):
        super(ReasoningLoss, self).__init__()

    def __call__(self, feat, feat2):
        new_feat = feat2.repeat_interleave(4, dim=0)
        loss = torch.norm(feat - new_feat, p=2)
        return loss
