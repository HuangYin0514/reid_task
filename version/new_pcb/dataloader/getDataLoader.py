import torch
import torchvision.transforms as T

from .datasets.market1501 import Market1501
from .utils.collate_batch import train_collate_fn, val_collate_fn
from .utils.RandomErasing import RandomErasing
from .utils.triplet_sampler import RandomIdentitySampler


def getData(opt):
    train_transforms = T.Compose(
        [
            T.Resize((opt.img_height, opt.img_width), interpolation=3),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            # RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = T.Compose(
        [
            T.Resize((opt.img_height, opt.img_width), interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # data loader
    train_dataset = Market1501(
        root=opt.dataset_path,
        data_folder="bounding_box_train",
        transform=train_transforms,
        relabel=True,
    )

    num_classes = train_dataset.num_pids

    query_dataset = Market1501(root=opt.dataset_path, data_folder="query", transform=test_transforms, relabel=False)
    gallery_dataset = Market1501(
        root=opt.dataset_path,
        data_folder="bounding_box_test",
        transform=test_transforms,
        relabel=False,
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     sampler=RandomIdentitySampler(train_dataset.dataset, opt.batch_size, num_instances=4),
    #     batch_size=opt.batch_size,
    #     collate_fn=train_collate_fn,
    # )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # sampler=RandomIdentitySampler(train_dataset.dataset, opt.batch_size, num_instances=4),
        batch_size=opt.batch_size,
        collate_fn=train_collate_fn,
    )

    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=opt.test_batch_size,
        shuffle=False,
        collate_fn=val_collate_fn,
    )
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=opt.test_batch_size,
        shuffle=False,
        collate_fn=val_collate_fn,
    )

    return train_loader, query_loader, gallery_loader, num_classes
