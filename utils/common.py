import errno
import importlib
import json
import os
import os.path as osp
import pickle
import random
import sys
import time
import warnings
from functools import wraps

import numpy as np
import PIL
import torch
from PIL import Image


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def read_config_file(config_file_path):
    spec = importlib.util.spec_from_file_location("config", config_file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def check_isfile(fpath):
    """Checks if the given path is a file.

    Args:
        fpath (str): file path.

    Returns:
       bool
    """
    isfile = osp.isfile(fpath)
    if not isfile:
        warnings.warn('No file found at "{}"'.format(fpath))
    return isfile


def tensors_to_numpy(*tensors):
    """
    Convert PyTorch tensors to NumPy arrays.

    Args:
        *tensors: Variable-length input tensors.

    Returns:
        tuple: A tuple containing the NumPy arrays corresponding to the input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].cpu().detach().numpy()

    np_arrays = tuple()
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor):
            # 如果是 PyTorch 张量，将其转换为 NumPy 数组
            np_arrays += (tensor.cpu().detach().numpy(),)
        elif isinstance(tensor, np.ndarray):
            # 如果已经是 NumPy 数组，不进行转换
            np_arrays += (tensor,)
        else:
            raise ValueError("Input must be either a PyTorch tensor or a NumPy array.")
    return np_arrays


class lazy_property:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        val = self.func(instance)
        setattr(instance, self.func.__name__, val)
        return val


def save_dict_info(dict_info, file_path):
    with open(file_path, "w") as f:
        for key, value in dict_info.items():
            f.write(f"{key}: {str(value)}\n")


def save_str_info(content, file_path):
    with open(file_path, "w") as f:
        f.write(content)


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t = time.time()
        result = func(*args, **kwargs)
        print("'" + func.__name__ + "'" + " took {:.2f} minute ".format((time.time() - t) / 60))
        return result

    return wrapper


def enable_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.enable_grad():
            result = func(*args, **kwargs)
        return result

    return wrapper


def deprecated(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # print('this function was deprecated!')
        raise Exception(f" {func.__name__} function was deprecated!")

    return wrapper


def to_pickle(thing, path):  # save something
    with open(path, "wb") as handle:
        pickle.dump(thing, handle, protocol=pickle.HIGHEST_PROTOCOL)


def from_pickle(path):  # load something
    thing = None
    with open(path, "rb") as handle:
        thing = pickle.load(handle)
    return thing


def pares_config(config, logger):
    keys_values_pairs = []  # List to store attribute-name and attribute-value pairs
    for attr_name in dir(config):
        if not attr_name.startswith("__"):  # Exclude private attributes
            attr_value = getattr(config, attr_name)  # Get the attribute value
            keys_values_pairs.append("{}: {}".format(attr_name, attr_value))  # Store the pair
    # Join the attribute-name and attribute-value pairs with newline separator
    full_output = "\t".join(keys_values_pairs)
    return full_output


def read_json(fpath):
    """Reads json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))