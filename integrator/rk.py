# -*- coding: utf-8 -*-
import numpy as np
import torch
from tqdm import tqdm

from .base import SolverBase


class RK4(SolverBase):
    def __init__(self, *args, **kwargs):
        super(RK4, self).__init__(*args, **kwargs)

    def step(self, func, t0, y0, dt, *args, **kwargs):
        k1 = dt * func(t0, y0)
        k2 = dt * func(t0 + 0.5 * dt, y0 + 0.5 * k1)
        k3 = dt * func(t0 + 0.5 * dt, y0 + 0.5 * k2)
        k4 = dt * func(t0 + dt, y0 + k3)
        y1 = y0 + (k1 + k2 * 2 + k3 * 2 + k4) / 6
        return y1
