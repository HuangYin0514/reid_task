import numpy as np
import torch


def _parse_data_for_eval(data):
    imgs, pids, camids = data
    return imgs, pids, camids


def _parse_data_for_eval_with_mask(data):
    imgs, masks, pids, camids = data
    return imgs, masks, pids, camids


def _extract_features(model, input):
    model.eval()
    return model(input)


def _extract_features_with_mask(model, img, mask):
    model.eval()
    return model(img, mask)


def feature_extract(data_loader, model, device):
    data_f, data_pids, data_camids = (
        [],
        [],
        [],
    )  # query features, query person IDs and query camera IDs
    for _, data in enumerate(data_loader):
        imgs, pids, camids = _parse_data_for_eval(data)
        imgs = imgs.to(device)
        features = _extract_features(model, imgs)
        data_f.append(features)
        data_pids.extend(pids)
        data_camids.extend(camids)
    data_f = torch.cat(data_f, 0)
    data_pids = np.asarray(data_pids)
    data_camids = np.asarray(data_camids)

    return data_f, data_pids, data_camids


def feature_extract_with_mask(data_loader, model, device):
    data_f, data_pids, data_camids = (
        [],
        [],
        [],
    )  # query features, query person IDs and query camera IDs
    for _, data in enumerate(data_loader):
        imgs, masks, pids, camids = _parse_data_for_eval_with_mask(data)
        imgs = imgs.to(device)
        masks = masks.to(device)
        features = _extract_features_with_mask(model, imgs, masks)
        data_f.append(features)
        data_pids.extend(pids)
        data_camids.extend(camids)
    data_f = torch.cat(data_f, 0)
    data_pids = np.asarray(data_pids)
    data_camids = np.asarray(data_camids)

    return data_f, data_pids, data_camids
