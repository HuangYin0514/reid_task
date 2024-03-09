import os.path as osp
import shutil

import cv2
import numpy as np

import utils

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results(distmat, dataset, data_type, width=128, height=256, save_dir="", topk=10):
    num_q, num_g = distmat.shape
    utils.common.mkdir_if_missing(save_dir)

    print("# query: {}\n# gallery {}".format(num_q, num_g))
    print("Visualizing top-{} ranks ...".format(topk))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat)[:, ::-1]

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == "gallery":
                suffix = "TRUE" if matched else "FALSE"
                dst = osp.join(dst, prefix + "_top" + str(rank).zfill(3)) + "_" + suffix
            else:
                dst = osp.join(dst, prefix + "_top" + str(rank).zfill(3))
            utils.common.mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + "_top" + str(rank).zfill(3) + "_name_" + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx][:3]
        qimg_path_name = qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path

        if data_type == "image":
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones(
                (
                    height,
                    num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING,
                    3,
                ),
                dtype=np.uint8,
            )
            grid_img[:, :width, :] = qimg
        else:
            qdir = osp.join(save_dir, osp.basename(osp.splitext(qimg_path_name)[0]))
            utils.common.mkdir_if_missing(qdir)
            _cp_img_to(qimg_path, qdir, rank=0, prefix="query")

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg_path, gpid, gcamid = gallery[g_idx][:3]
            invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = gpid == qpid
                if data_type == "image":
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color)
                    gimg = cv2.resize(gimg, (width, height))
                    start = rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (rank_idx + 1) * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                    grid_img[:, start:end, :] = gimg
                else:
                    _cp_img_to(
                        gimg_path,
                        qdir,
                        rank=rank_idx,
                        prefix="gallery",
                        matched=matched,
                    )

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == "image":
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname + ".jpg"), grid_img)

        if (q_idx + 1) % 100 == 0:
            print("- done {}/{}".format(q_idx + 1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))
