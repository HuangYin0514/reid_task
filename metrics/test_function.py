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

from metrics import distance, rank


def _parse_data_for_eval(data):
    imgs = data[0]
    pids = data[1]
    camids = data[2]
    return imgs, pids, camids


def _extract_features(model, input):
    model.eval()
    return model(input)


@torch.no_grad()
def test_function(model, test_loader, config, normalize_feature=False, dist_metric="cosine"):
    model.eval()

    # test dataloader------------------------------------------------------------
    query_dataloader, gallery_dataloader = test_loader

    # Extracting features from query set------------------------------------------------------------
    print("Extracting features from query set ...")
    qf, q_pids, q_camids = [], [], []  # query features, query person IDs and query camera IDs
    q_score = []
    for batch_idx, data in enumerate(query_dataloader):
        imgs, pids, camids = _parse_data_for_eval(data)
        imgs = imgs.to(config.device)
        features = _extract_features(model, imgs)
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Done, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    # Extracting features from gallery set------------------------------------------------------------
    print("Extracting features from gallery set ...")
    gf, g_pids, g_camids = [], [], []  # gallery features, gallery person IDs and gallery camera IDs
    g_score = []
    for batch_idx, data in enumerate(gallery_dataloader):
        imgs, pids, camids = _parse_data_for_eval(data)
        imgs = imgs.to(config.device)
        features = _extract_features(model, imgs)
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Done, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    # normalize_feature------------------------------------------------------------------------------
    if normalize_feature:
        print("Normalzing features with L2 norm ...")
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    # Computing distance matrix------------------------------------------------------------------------
    print("Computing distance matrix with metric={} ...".format(dist_metric))
    qf = np.array(qf.cpu())
    gf = np.array(gf.cpu())
    dist = distance.cosine_dist(qf, gf)
    rank_results = np.argsort(dist)[:, ::-1]

    # Computing CMC and mAP------------------------------------------------------------------------
    print("Computing CMC and mAP ...")
    APs, CMC = [], []
    for idx, data in enumerate(zip(rank_results, q_camids, q_pids)):
        a_rank, query_camid, query_pid = data
        ap, cmc = rank.compute_AP(a_rank, query_camid, query_pid, g_camids, g_pids)
        APs.append(ap), CMC.append(cmc)
    MAP = np.array(APs).mean()
    min_len = min([len(cmc) for cmc in CMC])
    CMC = [cmc[:min_len] for cmc in CMC]
    CMC = np.mean(np.array(CMC), axis=0)

    return CMC, MAP
