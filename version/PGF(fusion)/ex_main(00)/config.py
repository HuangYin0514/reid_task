import os

import numpy as np
import torch

########################################################################
# config.py
########################################################################
# For general settings
taskname = "PCB"
seed = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
dtype = torch.float32  # torch.float32 / torch.double

########################################################################
# For outputs settings
current_directory = os.path.dirname(os.path.realpath(__file__))
outputs_dir = "./outputs/"
outputs_path = os.path.join(current_directory, outputs_dir)

########################################################################
# For data settings
# Path setting
dataset_path = "/home/hy/project/reid/data/Market-1501-v15.09.15-test"
dataset_path = "/home/hy/project/reid/data/Market-1501-v15.09.15"
dataset_name = "market1501"

batch_size = 60
test_batch_size = 512

img_height = 256
img_width = 128

########################################################################
# For training settings
lr = 0.1

epochs = 6
print_every = 1
test_every = 2

epochs = 60
print_every = 1
test_every = 20
