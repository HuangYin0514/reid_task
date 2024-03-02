import numpy as np


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
