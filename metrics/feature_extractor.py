import numpy as np
import torch


def _parse_data_for_eval(data):
    imgs = data[0]
    pids = data[1]
    camids = data[2]

    return imgs, pids, camids


def _extract_features(model, input):
    model.eval()
    return model(input)


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
