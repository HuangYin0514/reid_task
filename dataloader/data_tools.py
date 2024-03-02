import sys
import os
import os.path as osp
import time
import errno
import json
from collections import OrderedDict
import warnings
import random
import numpy as np
import PIL
from PIL import Image

import torch


def read_image(path):
    """Reads image from path using ``PIL.Image``.
    Args:
        path (str): path to an image.
    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert("RGB")
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(path))
            pass
    return img


if __name__ == "__main__":
    i = read_image("data/Occluded_REID/occluded_body_images/002/002_01.tif1")
    print(i.size)
