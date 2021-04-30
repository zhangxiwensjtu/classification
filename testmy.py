import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

if __name__ == '__main__':
    # cell_training_loader = get_training_dataloader(
    #     path='/home/steadysjtu/cifar-10-batches-py',
    #     mean=settings.CIFAR100_TRAIN_MEAN,
    #     std=settings.CIFAR100_TRAIN_STD,
    #     num_workers=4,
    #     batch_size=1,
    #     shuffle=True
    # )
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=1,
        shuffle=True
    )
    for ii, (images, labels) in enumerate(cifar100_test_loader):
        if ii == 1:
            break
        print("images.shape = ", images.shape)
        print("labels=", labels.shape)