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

from . import distance, feature_extractor, rank, rerank


@torch.no_grad()
def test_function(
    model,
    q_loader,
    g_loader,
    normalize_feature=True,
    save_features=False,
    re_rank=False,
    eval_method="market1501",
    is_gpu=True,
    config=None,
    logger=None,
):
    model.eval()

    if is_gpu:
        device = config.device
    else:
        device = "cpu"
        model = model.to(device)
        logger.info("Using CPU ...")

    # Extracting features from query set(matrix size is qf.size(0), qf.size(1))
    print("Extracting features ...")
    qf, q_pids, q_camids = feature_extractor.feature_extract(q_loader, model, device)

    # Extracting features from gallery set(matrix size is gf.size(0), gf.size(1))
    gf, g_pids, g_camids = feature_extractor.feature_extract(g_loader, model, device)

    # Save feature
    if save_features:
        print("Save features ...")
        torch.save(qf, os.path.join(config.temps_outputs_path, "query_features_" + config.dataset_name + ".pt"))
        torch.save(gf, os.path.join(config.temps_outputs_path, "gallery_features_" + config.dataset_name + ".pt"))

    # Normalize feature
    if normalize_feature:
        print("Normalzing features with L2 norm ...")
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    # Computing distance matrix
    print("Computing distance matrix ...")
    distmat = distance.compute_distance_matrix(qf, gf).cpu().numpy()

    if re_rank:
        print("Applying person re-ranking ...")
        distmat_qq = distance.compute_distance_matrix(qf, qf).cpu().numpy()
        distmat_gg = distance.compute_distance_matrix(gf, gf).cpu().numpy()
        distmat = rerank.re_ranking(distmat, distmat_qq, distmat_gg)

    # Computing CMC and mAP
    if eval_method == "market1501":
        CMC, MAP = rank.eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50)

    return CMC, MAP


@torch.no_grad()
def test_function_with_mask(model, q_loader, g_loader, normalize_feature=True, save_features=True, config=None, logger=None):
    model.eval()

    device = config.device

    print("Extracting features ...")
    # Extracting features from query set(matrix size is qf.size(0), qf.size(1))
    qf, q_pids, q_camids = feature_extractor.feature_extract_with_mask(q_loader, model, device)
    # Extracting features from gallery set(matrix size is gf.size(0), gf.size(1))
    gf, g_pids, g_camids = feature_extractor.feature_extract_with_mask(g_loader, model, device)

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
