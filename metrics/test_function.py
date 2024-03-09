import argparse
import os
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(".")
sys.path.append(PARENT_DIR)

from . import distance, feature_extractor, rank


@torch.no_grad()
def test_function(model, q_loader, g_loader, normalize_feature=True, save_features=True, config=None, logger=None):
    model.eval()

    device = config.device

    # Extracting features from query set(matrix size is qf.size(0), qf.size(1))
    print("Extracting features ...")
    qf, q_pids, q_camids = feature_extractor.feature_extract(q_loader, model, device)

    # Extracting features from gallery set(matrix size is gf.size(0), gf.size(1))
    gf, g_pids, g_camids = feature_extractor.feature_extract(g_loader, model, device)

    # Save feature
    if save_features:
        print("Save features ...")
        torch.save(qf, os.path.join(config.outputs_path, "query_features_" + config.dataset_name + ".pt"))
        torch.save(gf, os.path.join(config.outputs_path, "gallery_features_" + config.dataset_name + ".pt"))

    # Normalize feature
    if normalize_feature:
        print("Normalzing features with L2 norm ...")
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    # Computing distance matrix
    print("Computing distance matrix ...")
    _, rank_results = distance.compute_distance_matrix(qf, gf)

    # Computing CMC and mAP
    CMC, MAP = rank.eval_rank(rank_results, q_camids, q_pids, g_camids, g_pids)

    return CMC, MAP
