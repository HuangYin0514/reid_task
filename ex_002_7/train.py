import argparse
import os
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(".")
sys.path.append(PARENT_DIR)

import model as model_function
from record import Recorder
from train_dataloader import getData

import loss_funciton
import metrics
import optim
import utils


@utils.common.timing
def brain(config, logger):
    logger.info("#" * 50)

    # Dataset
    train_loader, query_loader, gallery_loader, num_classes = getData(config=config)

    # Model
    model = model_function.ReidNet(num_classes=num_classes, config=config, logger=logger).to(config.device)

    # Loss function
    mse_loss = nn.MSELoss()
    ce_labelsmooth_loss = loss_funciton.CrossEntropyLabelSmoothLoss(num_classes=num_classes, config=config, logger=logger)
    ce_loss = nn.CrossEntropyLoss(reduction="mean")
    triplet_loss = loss_funciton.TripletLoss(margin=0.3)
    KLDivLoss = nn.KLDivLoss(reduction="batchmean")

    # Optimizer
    model_params_group = [{"params": model.parameters(), "lr": 0.00035, "weight_decay": 0.0005}]
    optimizer = torch.optim.Adam(model_params_group)

    # Scheduler
    scheduler = optim.WarmupMultiStepLR(
        optimizer,
        milestones=[40, 70],
        gamma=0.1,
        warmup_factor=0.01,
        warmup_iters=10,
        warmup_method="linear",
    )

    # Information record
    recorder = Recorder(config, logger)

    # Time
    start_time = time.time()

    # Train and Test
    for epoch in range(config.epochs):
        model.train()

        ## Train
        running_loss = 0.0
        for ind, data in enumerate(tqdm(train_loader)):
            ### data
            inputs, pids = data
            inputs = inputs.to(config.device)
            pids = pids.to(config.device)

            ### prediction
            optimizer.zero_grad()
            backbone_cls_score, backbone_pool_feats, backbone_bn_feats, resnet_feats, fusion_feats = model(inputs)

            ### Loss
            #### Gloab loss
            backbone_ce_loss = ce_labelsmooth_loss(backbone_cls_score, pids)
            backbone_tri_loss = triplet_loss(backbone_pool_feats, pids)
            backbone_loss = backbone_ce_loss + backbone_tri_loss

            #### Integrate loss
            integrate_feats, integrate_pids = model.integrate_feats_module(resnet_feats, pids, backbone_cls_score, num_same_id=4)
            integrate_cls_score = model.auxiliary_classifier_head(integrate_feats)
            integrate_ce_loss = ce_loss(integrate_cls_score, integrate_pids)

            #### Contrast loss
            contrast_mse_loss = mse_loss(fusion_feats, integrate_feats.repeat_interleave(4, dim=0))

            #### All loss
            loss = backbone_loss + 0.1 * integrate_ce_loss + 0.1 * contrast_mse_loss
            # print(backbone_loss.item(), 0.1 * integrate_ce_loss.item(), 0.1 * contrast_mse_loss.item())

            ### Update the parameters
            loss.backward()
            optimizer.step()

            ### record Loss
            running_loss += loss.item() * inputs.size(0)

        scheduler.step()

        ## Logger
        if epoch % config.print_every == 0:
            ### Log message
            epoch_loss = running_loss / len(train_loader.dataset)
            time_remaining = (config.epochs - epoch) * (time.time() - start_time) / (epoch + 1)
            time_remaining_H = time_remaining // 3600
            time_remaining_M = time_remaining / 60 % 60
            message = ("Epoch {0}/{1}\t" "Training Loss: {epoch_loss:.4f}\t" "Time remaining is {time_H:.0f}h:{time_M:.0f}m").format(epoch + 1, config.epochs, epoch_loss=epoch_loss, time_H=time_remaining_H, time_M=time_remaining_M)
            logger.info(message)

            ### Record train information
            recorder.train_epochs_list.append(epoch + 1)
            recorder.train_loss_list.append(epoch_loss)

        ## Test
        condition1 = epoch + 1 == config.epochs
        condition2 = (epoch + 1) >= config.epoch_start_test and (epoch + 1) % config.test_every == 0
        if condition1 or condition2:
            ### Test datset
            torch.cuda.empty_cache()
            CMC, mAP = metrics.test_function(model, query_loader, gallery_loader, config=config, logger=logger)

            ### Log test information
            message = ("Epoch {}/{}\t" "Testing: dataset_name: {} top1: {:.3f} top5: {:.3f} top10: {:.3f} mAP: {:.3f}").format(epoch + 1, config.epochs, config.dataset_name, CMC[0] * 100, CMC[4] * 100, CMC[9] * 100, mAP * 100)
            logger.info(message)

            ### Save model
            model_path = os.path.join(config.models_outputs_path, "model_{}.tar".format(epoch + 1))
            # torch.save(model.state_dict(), model_path)

            ### Record test information
            recorder.val_epochs_list.append(epoch + 1)
            recorder.val_CMC_list.append(CMC)
            recorder.val_mAP_list.append(mAP)

        recorder.save()


if __name__ == "__main__":
    ######################################################################
    #
    # environment
    #
    ######################################################################
    # Config
    parser = argparse.ArgumentParser(description=None)  ## Parse command-line arguments
    parser.add_argument("--config_file", type=str, help="Path to the config.py file")
    parser.add_argument("--dataset_path", type=str, help="Path to the Dataset")
    parser.add_argument("--some_float", type=float, default=0.0, help="")
    parser.add_argument("--some_int", type=int, default=0, help="")
    args = parser.parse_args()
    config = utils.common.read_config_file(args.config_file)  ## Read the configuration from the provided file
    # config.some_float = args.some_float ## Set command-line to config

    config.dataset_path = args.dataset_path  ## Set command-line to config

    # Directory
    ## Set up the dataset directory
    dataset_path = config.dataset_path
    if not os.path.exists(dataset_path):
        raise RuntimeError("Dataset path does not exist!")
    ## Set up the outputs directory
    outputs_path = config.outputs_path
    if os.path.exists(outputs_path):
        shutil.rmtree(outputs_path)
    utils.common.mkdir_if_missing(config.models_outputs_path)
    utils.common.mkdir_if_missing(config.logs_outputs_path)
    utils.common.mkdir_if_missing(config.temps_outputs_path)

    # Initialize a logger tool
    logger = utils.logger.Logger(config.logs_outputs_path)
    logger.info("#" * 50)
    logger.info("Config values: {}".format(utils.common.pares_config(config, logger)))
    logger.info(f"Task: {config.taskname}")
    logger.info(f"Using device: {config.device}")
    logger.info(f"Using data type: {config.dtype}")

    # Set environment
    random.seed(config.seed)
    os.environ["PYTHONASHSEED"] = str(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # The result cannot be reproduced when True

    # Set device
    if config.device == "cuda":
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"GPU is_available: {torch.cuda.is_available()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Current device id: {torch.cuda.current_device()}")
    else:
        raise RuntimeError("Unsupported device for cpu!")

    # Training
    start_time = time.time()
    brain(config, logger)
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info("The running time of training: {:.3} h".format(execution_time / 3600))
