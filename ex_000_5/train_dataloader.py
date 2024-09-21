import torch
import torchvision.transforms as T

import data_function


def getData(config):

    # Transforms
    train_transforms = T.Compose(
        [
            T.Resize((config.img_height, config.img_width), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop((config.img_height, config.img_width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            data_function.transforms.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
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
    dataset = data_function.datasets.DukeMTMCreID(root=config.dataset_path)
    num_classes = dataset.num_train_pids

    # Dataloder
    ## Train dataloder
    train_set = data_function.ImageDataset(dataset.train, train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=data_function.samplers.RandomIdentitySampler(dataset.train, config.batch_size, num_instances=4),
        batch_size=config.batch_size,
        collate_fn=data_function.train_collate_fn,
    )

    ## Query dataloder
    query_set = data_function.ImageDataset(dataset.query, test_transforms)
    query_loader = torch.utils.data.DataLoader(query_set, batch_size=config.test_batch_size, shuffle=False, collate_fn=data_function.val_collate_fn)

    ## Gallery dataloder
    gallery_set = data_function.ImageDataset(dataset.gallery, test_transforms)
    gallery_loader = torch.utils.data.DataLoader(gallery_set, batch_size=config.test_batch_size, shuffle=False, collate_fn=data_function.val_collate_fn)

    return train_loader, query_loader, gallery_loader, num_classes
