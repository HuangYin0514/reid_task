import torch
import torch.nn.functional as F
from torch import nn


class IntegratFeatureMap:
    def __init__(self, config):
        super(IntegratFeatureMap, self).__init__()
        self.config = config

    def __call__(self, feats_map, pids):
        bs = feats_map.size(0)
        c, h, w = feats_map.size(1), feats_map.size(2), feats_map.size(3)
        chunk_size = int(bs / 4)  # 15

        integrat_feats_map = torch.zeros([chunk_size, c, h, w]).cuda()
        integrat_pids = torch.zeros([chunk_size]).to(torch.int64)
        for i in range(chunk_size):
            integrat_feats_map[i] = feats_map[4 * i] + feats_map[4 * i + 1] + feats_map[4 * i + 2] + feats_map[4 * i + 3]
            integrat_pids[i] = pids[4 * i]

        return integrat_feats_map, integrat_pids
