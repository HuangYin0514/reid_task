import random
import time
from functools import wraps

import numpy as np
import torch


# 求导
def dfx(f, x):
    return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), retain_graph=True, create_graph=True)[0]
