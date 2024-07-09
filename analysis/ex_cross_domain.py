import glob
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

current_dir = os.path.dirname(os.path.abspath("__file__"))  # 当前目录
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))  # 上一级目录
sys.path.append(parent_dir)
grandparent_dir = os.path.abspath(os.path.join(parent_dir, ".."))  # 上两级目录
sys.path.append(grandparent_dir)

import data_function
import metrics
import tools
import utils
from ex_main.model import *
from ex_main.train_dataloader import getData
from utils.config_plot import *

###################################################################################################
ppath = "/home/hy/project/reid_task/ex_main/"

# Config
config_file_path = os.path.join(ppath, "config.py")
config = utils.common.read_config_file(config_file_path)

# Outputs path
log_path = os.path.join(config.outputs_path, "train_log.log")
if os.path.exists(log_path):
    os.remove(log_path)

# Initialize a logger tool
logger = utils.logger.Logger(config.outputs_path)
logger.info("#" * 50)
logger.info(f"Task: {config.taskname}")
logger.info(f"Using device: {config.device}")
logger.info(f"Using data type: {config.dtype}")

# Set environment
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # The result cannot be reproduced when True

# Outputs path
vis_outputs_path = "./results/vis/"
if os.path.exists(vis_outputs_path):
    shutil.rmtree(vis_outputs_path)

###################################################################################################
# Data
config.test_batch_size = 512
_, query_loader, gallery_loader, num_classes = getData(config=config)


filePath = os.path.join(ppath, "outputs/models/")
for root, dirs, files in os.walk(filePath):
    for file in files:
        logger.info("=" * 100)
        model_file = os.path.join(root, file)
        logger.info("Model file: {}".format(model_file))
        # Model
        model = ReidNet(num_classes=num_classes, config=config, logger=logger).to(config.device)
        path = model_file
        utils.network.load_network(model, path, config.device)
        logger.info("Model numbers of parameters: {:.2f}M".format(utils.network.count_parameters(model) / 1e6))
        logger.info("Model numbers of parameters: {}".format(utils.network.count_parameters(model)))

        # Infer of function
        re_rank = False  # True / False
        CMC, mAP = metrics.test_function(model, query_loader, gallery_loader, re_rank=re_rank, config=config, logger=logger)
        message = ("Testing: dataset_name: {} top1:{:.4f} top5:{:.4f} top10:{:.4f} mAP:{:.4f}").format(config.dataset_name, CMC[0], CMC[4], CMC[9], mAP)
        logger.info(message)
###################################################################################################
