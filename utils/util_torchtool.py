
from collections import OrderedDict
import shutil
import warnings
import os
import os.path as osp
from functools import partial
import pickle

import torch
import torch.nn as nn


# open_specified_layers-----------------------------------------------------------
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


# open_all_layers-----------------------------------------------------------
def open_all_layers(model):
    r"""Opens all layers in model for training.
    Examples::
        >>> from torchreid.utils import open_all_layers
        >>> open_all_layers(model)
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True

# weights_init_kaiming----------------------------------------------------------


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.constant_(m.weight.data, 1)
        torch.nn.init.constant_(m.bias.data, 0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, std=0.001)
        torch.nn.init.constant_(m.bias.data, 0.0)
