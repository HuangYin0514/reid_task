import os
import os.path as osp
import pickle
import shutil
import warnings
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_network(network, file_path, device):
    if not os.path.exists(file_path):
        raise RuntimeError("Cannot find pretrained network at '{}'".format(file_path))

    # Original saved file with DataParallel
    state_dict = torch.load(file_path, map_location=torch.device(device))

    # state dict
    model_dict = network.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    # load model state ---->{matched_layers, discarded_layers}
    for k, v in state_dict.items():
        if k.startswith("module."):
            print("discarded_layers {}".format(k[:8]))
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    network.load_state_dict(model_dict)

    # assert model state
    if len(matched_layers) == 0:
        warnings.warn('The pretrained weights "{}" cannot be loaded, ' "please check the key names manually " "(** ignored and continue **)".format(matched_layers))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(file_path))
        if len(discarded_layers) > 0:
            print("** The following layers are discarded " "due to unmatched keys or layer size: {}".format(discarded_layers))

    return network


def open_specified_layers(model, open_layers):
    r"""Opens specified layers in model for training while keeping
    other layers frozen.
    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.
    Examples::
        >>> from torchreid.utils import open_specified_layers
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(model, layer), '"{}" is not an attribute of the model, please provide the correct name'.format(layer)

    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def open_all_layers(model):
    r"""Opens all layers in model for training.
    Examples::
        >>> from torchreid.utils import open_all_layers
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True
