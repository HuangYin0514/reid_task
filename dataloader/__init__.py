import torch
from dataloader.datasets import dataset_loader
from torchvision import datasets

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import ImageDataset, init_dataset
from .samplers import (  # New add by gu
    RandomIdentitySampler,
    RandomIdentitySampler_alignedreid,
)
from .transforms import build_transforms


def getDataLoader(dataset_name, dataset_path, config):

    num_workers = 4  # Resulting in the inability to reproduce results

    # transforms  --------------------------------------------------------
    train_transforms = build_transforms(config, is_train=True)
    val_transforms = build_transforms(config, is_train=False)

    # dataset ------------------------------------------------------------
    dataset = init_dataset(dataset_name, root=dataset_path)
    train_set = ImageDataset(dataset.train, train_transforms)
    num_classes = dataset.num_train_pids

    # train_loader ------------------------------------------------------------
    train_loader = None
    if config.data_sampler_type == "softmax":
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, collate_fn=train_collate_fn)
    elif config.data_sampler_type == "RandomIdentitySampler":
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, sampler=RandomIdentitySampler(dataset.train, config.batch_size, config.num_instance), num_workers=num_workers, collate_fn=train_collate_fn)
    else:
        raise RuntimeError("data_sampler_type of {} not exits".format(config.data_sampler_type))

    # query_set ------------------------------------------------------------
    query_set = ImageDataset(dataset.query, val_transforms)
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=config.test_batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn)

    # gallery_set ------------------------------------------------------------
    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=config.test_batch_size, shuffle=False, num_workers=num_workers, collate_fn=val_collate_fn)
    return train_loader, query_loader, gallery_loader, num_classes


# def check_data(images, fids, img_save_path):
#     """
#     check data of image
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import torchvision.utils as vutils

#     # [weight, hight]
#     plt.figure(figsize=(10, 10))
#     plt.axis("off")
#     plt.title(fids)
#     plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=True), (1, 2, 0)))
#     plt.savefig(img_save_path)
