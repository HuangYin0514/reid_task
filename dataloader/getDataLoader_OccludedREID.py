import torchvision.transforms as T
import torch
from .utils.collate_batch import val_collate_fn
from .datasets.occluded_reid import Occluded_REID


def getOccludedData(opt, data_dir):
    test_transforms = T.Compose(
        [
            T.Resize((opt.img_height, opt.img_width), interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # data loader
    query_dataset = Occluded_REID(
        root=data_dir,
        data_folder="occluded_body_images",
        transform=test_transforms,
        relabel=False,
        is_query=True,
    )
    gallery_dataset = Occluded_REID(
        root=data_dir,
        data_folder="whole_body_images",
        transform=test_transforms,
        relabel=False,
        is_query=False,
    )

    query_loader = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=val_collate_fn,
    )
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=opt.test_batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        collate_fn=val_collate_fn,
    )

    return query_loader, gallery_loader
