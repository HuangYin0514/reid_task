import numpy as np


def in1d(array1, array2, invert=False):
    """
    :param set1: np.array, 1d
    :param set2: np.array, 1d
    :return:
    """
    mask = np.in1d(array1, array2, invert=invert)
    return array1[mask]


def notin1d(array1, array2):
    return in1d(array1, array2, invert=True)


def compute_AP(a_rank, query_camid, query_pid, gallery_camids, gallery_pids, mode="inter-camera"):
    """given a query and all galleries, compute its ap and cmc"""

    if mode == "inter-camera":
        junk_index_1 = in1d(np.argwhere(query_pid == gallery_pids), np.argwhere(query_camid == gallery_camids))
        junk_index_2 = np.argwhere(gallery_pids == -1)
        junk_index = np.append(junk_index_1, junk_index_2)
        index_wo_junk = notin1d(a_rank, junk_index)
        good_index = in1d(np.argwhere(query_pid == gallery_pids), np.argwhere(query_camid != gallery_camids))
    elif mode == "intra-camera":
        junk_index_1 = np.argwhere(query_camid != gallery_camids)
        junk_index_2 = np.argwhere(gallery_pids == -1)
        junk_index = np.append(junk_index_1, junk_index_2)
        index_wo_junk = notin1d(a_rank, junk_index)
        good_index = np.argwhere(query_pid == gallery_pids)
    elif mode == "all":
        junk_index = np.argwhere(gallery_pids == -1)
        index_wo_junk = notin1d(a_rank, junk_index)
        good_index = in1d(np.argwhere(query_pid == gallery_pids))

    num_good = len(good_index)
    hit = np.in1d(index_wo_junk, good_index)
    index_hit = np.argwhere(hit == True).flatten()
    if len(index_hit) == 0:
        AP = 0
        cmc = np.zeros([len(index_wo_junk)])
    else:
        precision = []
        for i in range(num_good):
            precision.append(float(i + 1) / float((index_hit[i] + 1)))
        AP = np.mean(np.array(precision))
        cmc = np.zeros([len(index_wo_junk)])
        cmc[index_hit[0] :] = 1
    return AP, cmc


def eval_rank(rank_results, q_camids, q_pids, g_camids, g_pids):

    APs, CMC = [], []
    for _, data in enumerate(zip(rank_results, q_camids, q_pids)):
        a_rank, query_camid, query_pid = data
        ap, cmc = compute_AP(a_rank, query_camid, query_pid, g_camids, g_pids)
        APs.append(ap), CMC.append(cmc)
    MAP = np.array(APs).mean()
    min_len = min([len(cmc) for cmc in CMC])
    CMC = [cmc[:min_len] for cmc in CMC]
    CMC = np.mean(np.array(CMC), axis=0)

    return CMC, MAP


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
