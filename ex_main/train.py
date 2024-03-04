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

from dataloader.getDataLoader import getData
from loss.crossentropy_labelsmooth_loss import CrossEntropyLabelSmoothLoss
from loss.triplet_loss import TripletLoss
from metrics import distance, rank
from metrics.test_function import test_function
from model import pcb_ffm
from record import Recorder
from utils import (
    Logger,
    count_parameters,
    read_config_file,
    save_config,
    timing,
    to_pickle,
)


@timing
def brain(config, logger):
    logger.info("#" * 50)

    # Dataset
    train_loader, query_loader, gallery_loader, num_classes = getData(opt=config)
    test_loader, test_query_loader, test_gallery_loader, test_num_classes = getData(opt=config)

    val_loader = [query_loader, gallery_loader]
    test_loader = [test_query_loader, test_gallery_loader]

    # Model
    model = pcb_ffm(num_classes=num_classes, height=config.img_height, width=config.img_width).to(config.device)

    # Loss function
    ce_labelsmooth_loss = CrossEntropyLabelSmoothLoss(num_classes=num_classes, config=config, logger=logger)
    triplet_loss = TripletLoss(margin=0.3)

    # Optimizer
    base_param_ids = set(map(id, model.backbone.parameters()))
    new_params = [p for p in model.parameters() if id(p) not in base_param_ids]
    param_groups = [{"params": model.backbone.parameters(), "lr": config.lr / 10}, {"params": new_params, "lr": config.lr}]
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Information record
    recorder = Recorder(config, logger)

    # Time
    start_time = time.time()

    # Train and Test
    for epoch in range(config.epochs):
        model.train()
        scheduler.step()

        ## Train
        running_loss = 0.0
        for ind, data in enumerate(tqdm(train_loader)):
            ### data
            inputs, labels = data
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            ### prediction
            optimizer.zero_grad()
            parts_scores, gloab_features, fusion_feature = model(inputs)

            ### Loss
            #### Gloab loss
            gloab_loss = triplet_loss(gloab_features, labels)

            #### Fusion loss
            fusion_loss = triplet_loss(fusion_feature, labels)

            #### Parts loss
            part_loss = 0
            for logits in parts_scores:
                stripe_loss = ce_labelsmooth_loss(logits, labels)
                part_loss += stripe_loss

            #### all of loss
            loss_alph = 1
            loss_beta = 0.01
            loss = 0.1 * part_loss + loss_alph * gloab_loss[0] + loss_beta * fusion_loss[0]

            ### Update the parameters
            loss.backward()
            optimizer.step()

            ### record Loss
            running_loss += loss.item() * inputs.size(0)

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
        if (epoch + 1) % config.test_every == 0 or epoch + 1 == config.epochs:
            ### Test datset
            torch.cuda.empty_cache()
            CMC, mAP = test_function(model, val_loader, config)

            ### Log test information
            message = ("Testing: dataset_path: {} top1:{:.4f} top5:{:.4f} top10:{:.4f} mAP:{:.4f}").format(config.dataset_path, CMC[0], CMC[4], CMC[9], mAP)
            logger.info(message)

            ### Save model
            model_path = os.path.join(config.outputs_path, "model_{}.tar".format(epoch + 1))
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
    ## Parse command-line arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--config_file", type=str, help="Path to the config.py file")
    parser.add_argument("--some_float", type=float, default=0.0, help="")
    parser.add_argument("--some_int", type=int, default=0, help="")
    args = parser.parse_args()
    ## Read the configuration from the provided file
    config_file_path = args.config_file
    config = read_config_file(config_file_path)
    ## Set command-line to config
    ## config.some_float = args.some_float

    # Directory
    ## Set up the dataset directory
    dataset_path = config.dataset_path
    if not os.path.exists(dataset_path):
        raise RuntimeError("Dataset path does not exist!")
    ## Set up the outputs directory
    outputs_path = config.outputs_path
    if os.path.exists(outputs_path):
        shutil.rmtree(outputs_path)
    os.makedirs(outputs_path)

    # Initialize a logger tool
    logger = Logger(outputs_path)
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

    # Set device
    if config.device == "cuda":
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"GPU is_available: {torch.cuda.is_available()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Current device id: {torch.cuda.current_device()}")
    else:
        # raise Exception("Unsupported device for cpu!")
        logger.info("warining using CPU!" * 100)

    # Training
    try:
        start_time = time.time()
        brain(config, logger)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("The running time of training: {:.5e} s".format(execution_time))

    except Exception as e:
        logger.error(traceback.format_exc())
        logger.info("An error occurred: {}".format(e))

    # Logs all the attributes and their values present in the given config object.
    save_config(config, logger)
