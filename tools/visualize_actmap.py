"""Visualizes CNN activation maps to see where the CNN focuses on to extract features.

Reference:
    - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
      performance of convolutional neural networks via attention transfer. ICLR, 2017
    - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
"""

import argparse
import os.path as osp

import cv2
import numpy as np
import torch
from torch.nn import functional as F

import utils

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(model, test_loader, save_dir, width, height, use_gpu, img_mean=None, img_std=None):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()

    for target in list(["dataset"]):
        # data_loader = test_loader[target]["query"]  # only process query images
        # original images and activation maps are saved individually
        actmap_dir = osp.join(save_dir, "actmap_" + target)
        utils.common.mkdir_if_missing(actmap_dir)
        print("Visualizing activation maps for {} ...".format(target))

        for batch_idx, data in enumerate(test_loader):

            imgs, pids, camids, paths = data

            if use_gpu:
                imgs = imgs.cuda()

            # forward to get convolutional feature maps
            try:
                outputs = model.heatmap(imgs)
            except TypeError:
                raise TypeError("heatmap got unexpected keyword method, return feature maps only.")

            if outputs.dim() != 4:
                raise ValueError(
                    "The model output is supposed to have " "shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. " "Please make sure you set the model output at eval mode " "to be the last convolutional feature maps".format(outputs.dim())
                )

            # compute activation maps
            # outputs = (outputs**2).sum(1)

            # outputs = torch.abs(outputs)
            # outputs = outputs.sum(1)

            outputs = torch.abs(outputs)
            outputs = torch.max(outputs, dim=1, keepdim=True)[0]
            outputs = outputs.squeeze_(1)

            b, h, w = outputs.size()
            outputs = outputs.view(b, h * w)
            outputs = F.normalize(outputs, p=2, dim=1)
            outputs = outputs.view(b, h, w)

            if use_gpu:
                imgs, outputs = imgs.cpu(), outputs.cpu()

            for j in range(outputs.size(0)):
                # get image name
                path = paths[j]
                imname = osp.basename(osp.splitext(path)[0])

                # RGB image
                img = imgs[j, ...]
                for t, m, s in zip(img, img_mean, img_std):
                    t.mul_(s).add_(m).clamp_(0, 1)
                img_np = np.uint8(np.floor(img.numpy() * 255))
                img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

                # activation map
                outputs[j, :2, :] = 0
                outputs[j, -2:, :] = 0
                outputs[j, :, :2] = 0
                outputs[j, :, -2:] = 0
                am = outputs[j, ...].numpy()
                # am = outputs[j, 2:-2:, 2:-2].numpy()
                am = cv2.resize(am, (width, height))
                am = 255 * (am - np.min(am)) / (np.max(am) - np.min(am) + 1e-12)
                am = np.uint8(np.floor(am))
                am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

                # return img_np, am, outputs

                # overlapped
                overlapped = img_np * 0.5 + am * 0.5
                overlapped[overlapped > 255] = 255
                overlapped = overlapped.astype(np.uint8)

                # save images in a single figure (add white spacing between images)
                # from left to right: original image, activation map, overlapped image
                grid_img = 255 * np.ones((height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8)
                grid_img[:, :width, :] = img_np[:, :, ::-1]
                grid_img[:, width + GRID_SPACING : 2 * width + GRID_SPACING, :] = am
                grid_img[:, 2 * width + 2 * GRID_SPACING :, :] = overlapped
                cv2.imwrite(osp.join(actmap_dir, imname + ".jpg"), grid_img)

            if (batch_idx + 1) % 10 == 0:
                print("- done batch {}/{}".format(batch_idx + 1, len(test_loader)))
