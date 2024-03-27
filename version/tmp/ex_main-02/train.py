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
from tqdm import tqdm

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(".")
sys.path.append(PARENT_DIR)

from model import *
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
    model = ReidNet(num_classes=num_classes, config=config, logger=logger).to(config.device)

    # Loss function
    ce_labelsmooth_loss = loss_funciton.CrossEntropyLabelSmoothLoss(num_classes=num_classes, config=config, logger=logger)
    triplet_loss = loss_funciton.TripletLoss(margin=0.3)
    center_loss = loss_funciton.CenterLoss(num_classes=num_classes, feature_dim=2048, config=config, logger=logger)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.00035,
        weight_decay=0.0005,
    )

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
            inputs, labels = data
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            ### prediction
            optimizer.zero_grad()
            gloab_score, gloab_feat, ode_score, ode_feat = model(inputs)

            ### Loss
            #### Gloab loss
            gloab_ce_loss = ce_labelsmooth_loss(gloab_score, labels)
            gloab_tri_loss = triplet_loss(gloab_feat, labels)
            gloab_cent_loss = center_loss(gloab_feat, labels)
            gloab_loss = gloab_ce_loss + gloab_tri_loss + 0.0005 * gloab_cent_loss

            #### ODEnet loss
            ode_ce_loss = ce_labelsmooth_loss(ode_score, labels)
            ode_loss = ode_ce_loss

            #### All loss
            loss = gloab_loss + 0.1 * ode_loss

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
            message = ("Testing: dataset_name: {} top1:{:.4f} top5:{:.4f} top10:{:.4f} mAP:{:.4f}").format(config.dataset_name, CMC[0], CMC[4], CMC[9], mAP)
            logger.info(message)

            ### Save model
            model_path = os.path.join(config.models_outputs_path, "model_{}.tar".format(epoch + 1))
            torch.save(model.state_dict(), model_path)

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
    parser.add_argument("--some_float", type=float, default=0.0, help="")
    parser.add_argument("--some_int", type=int, default=0, help="")
    args = parser.parse_args()
    config = utils.common.read_config_file(args.config_file)  ## Read the configuration from the provided file
    # config.some_float = args.some_float ## Set command-line to config

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
    try:
        start_time = time.time()
        brain(config, logger)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("The running time of training: {:.5e} s".format(execution_time))
    except Exception as e:
        logger.info("An error occurred: {}".format(e))
        raise RuntimeError(traceback.format_exc())
