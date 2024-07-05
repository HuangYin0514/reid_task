import os

import numpy as np
import torch

########################################################################
# config.py
########################################################################
# For general settings
taskname = "ReID_Task"
seed = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # torch.float32 / torch.double

########################################################################
# For outputs settings
current_directory = os.path.dirname(os.path.realpath(__file__))
outputs_dir = "./outputs/"
outputs_path = os.path.join(current_directory, outputs_dir)
models_outputs_path = os.path.join(outputs_path, "models")
logs_outputs_path = os.path.join(outputs_path, "logs")
temps_outputs_path = os.path.join(outputs_path, "temps")

########################################################################
# For data settings
batch_size = 60
test_batch_size = 256
img_height = 384
img_width = 128

# Path setting
dataset_name = "market1501"
dataset_path = "/home/hy/project/data/Market-1501-v15.09.15"
########################################################################
# For training settings
epochs = 16000
print_every = 1
test_every = 5
epoch_start_test = 90

# ########################################################################
# # Test
# dataset_path = "/home/hy/project/data/Market-1501-v15.09.15-test"
# epochs = 1
# test_every = 1
# epoch_start_test = 0
