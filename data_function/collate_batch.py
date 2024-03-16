# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    (imgs, pids, camids, impath) = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    (imgs, pids, camids, impath) = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids


def train_collate_with_mask_fn(batch):
    (imgs, pids, camids, impath, masks) = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(masks, dim=0), pids


def val_collate_with_mask_fn(batch):
    (imgs, pids, camids, impath, masks) = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), torch.stack(masks, dim=0), pids, camids
