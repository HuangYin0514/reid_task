import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import utils


class Recorder:
    def __init__(self, config, logger, **kwargs):
        super(Recorder, self).__init__()

        self.config = config
        self.logger = logger

        self.train_epochs_list = []
        self.train_loss_list = []

        self.val_epochs_list = []
        self.val_CMC_list = []
        self.val_mAP_list = []

    def plot_results_fig(self):
        train_epochs_list = self.train_epochs_list
        train_loss_list = self.train_loss_list

        val_epochs_list = self.val_epochs_list
        val_CMC_list = np.array(self.val_CMC_list)
        val_mAP_list = np.array(self.val_mAP_list)

        if len(val_CMC_list) == 0:
            return

        # Plot results
        num_lines = 1
        num_rows = 2
        fig, axs = plt.subplots(num_lines, num_rows, figsize=(4 * num_rows, 3 * num_lines), dpi=100)

        ## Plot train loss
        subfig = axs[0]
        subfig.set_title("(a)")
        subfig.set_xlabel("Iterations")
        subfig.set_ylabel("Train loss")
        subfig.plot(train_epochs_list, train_loss_list, label="Train loss")
        ### text point
        # subfig.axvline(
        #     train_epochs_list[np.argmin(train_loss_list)],
        #     c="r",
        #     ls="--",
        # )
        subfig.text(
            train_epochs_list[np.argmin(train_loss_list)] * 0.4,
            np.min(train_loss_list) * 1.2,
            "Train loss in {}, {:.2f}".format(train_epochs_list[np.argmin(train_loss_list)], np.min(train_loss_list)),
            color="g",
        )
        subfig.set_yscale("log")
        subfig.legend()

        ## Plot metrics
        subfig = axs[1]
        subfig.set_title("(b)")
        subfig.set_xlabel("Iterations")
        subfig.set_ylabel("Metrics")
        subfig.plot(val_epochs_list, val_CMC_list[:, 0], label="Rank-1")
        subfig.plot(val_epochs_list, val_mAP_list, label="mAP")
        subfig.plot(
            val_epochs_list[np.argmax(val_CMC_list[:, 0])],
            np.max(val_CMC_list[:, 0]),
            marker="o",
            color="g",
        )
        subfig.plot(
            val_epochs_list[np.argmax(val_mAP_list)],
            np.max(val_mAP_list),
            marker="o",
            color="g",
        )
        ### text point
        tmp_index = np.argmax(val_CMC_list[:, 0])
        tmp_epoch = val_epochs_list[tmp_index]
        tmp_R1 = val_CMC_list[tmp_index, 0]
        tmp_mAP = val_mAP_list[tmp_index]
        subfig.text(
            tmp_epoch * 0.55,
            tmp_R1 * 0.98,
            "R@1 in {}, {:.2f}%/{:.2f}%".format(
                tmp_epoch,
                tmp_R1 * 100,
                tmp_mAP * 100,
            ),
            color="g",
        )
        tmp_index = np.argmax(val_mAP_list)
        tmp_epoch = val_epochs_list[tmp_index]
        tmp_R1 = val_CMC_list[tmp_index, 0]
        tmp_mAP = val_mAP_list[tmp_index]
        subfig.text(
            tmp_epoch * 0.55,
            tmp_mAP * 1.02,
            "mAP in {}, {:.2f}%/{:.2f}%".format(
                tmp_epoch,
                tmp_R1 * 100,
                tmp_mAP * 100,
            ),
            color="g",
        )
        # subfig.set_yscale("log")
        subfig.legend()
        plt.tight_layout()

        path = os.path.join(self.config.logs_outputs_path, "loss_curve.png")
        plt.savefig(path)
        plt.close(fig)

    def save(self):
        stats = {}

        stats["train_epochs_list"] = self.train_epochs_list
        stats["train_loss_list"] = self.train_loss_list

        stats["val_epochs_list"] = self.val_epochs_list
        stats["val_CMC_list"] = self.val_CMC_list
        stats["val_mAP_list"] = self.val_mAP_list

        filename = f"recorder.pkl"
        path = os.path.join(self.config.temps_outputs_path, filename)
        utils.common.to_pickle(stats, path)

        self.plot_results_fig()
