import torch
import torchvision.transforms as T

import dataloader


def getData(config):
    num_workers = 4  # Resulting in the inability to reproduce results

    # Transforms
    train_transforms = T.Compose(
        [
            T.Resize((config.img_height, config.img_width), interpolation=3),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            # RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = T.Compose(
        [
            T.Resize((config.img_height, config.img_width), interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Dataset
    dataset = dataloader.datasets.Market1501(root=config.dataset_path)
    num_classes = dataset.num_train_pids

    # Dataloder
    ## Train dataloder
    train_set = dataloader.ImageDataset(dataset.train, train_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, sampler=dataloader.samplers.RandomIdentitySampler(train_set.dataset, config.batch_size, num_instances=4), batch_size=config.batch_size, collate_fn=dataloader.train_collate_fn)

    ## Query dataloder
    query_set = dataloader.ImageDataset(dataset.query, test_transforms)
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=config.test_batch_size, shuffle=False, collate_fn=dataloader.val_collate_fn)

    ## Gallery dataloder
    gallery_set = dataloader.ImageDataset(dataset.gallery, test_transforms)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=config.test_batch_size, shuffle=False, collate_fn=dataloader.val_collate_fn)

    return train_loader, query_loader, gallery_loader, num_classes
