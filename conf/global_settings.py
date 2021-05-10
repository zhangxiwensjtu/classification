""" configurations for this project

author baiyu
"""
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
# cell_train_mean = (0.76666977, 0.52926302, 0.68443703)  # 这个是我测的
# cell_train_std = (0.13385421, 0.29181167, 0.21264157)
cell_train_mean = [0.644702, 0.5087477, 0.7866369]  # 这个是用现成的代码测的
cell_train_std = [0.24641053, 0.31298763, 0.14136614]

# cell_train_mean = [0.485, 0.456, 0.406]  # 这个是一般普遍用的
# cell_train_std = [0.229, 0.224, 0.225]


# CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
# CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = '/home/steadysjtu/classification/checkpoint'

#total training epoches
# EPOCH = 200
EPOCH = 300
MILESTONES = [100, 200, 300]
# MILESTONES = 20000
# MILESTONES = [60, 120, 180]
#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = '/home/steadysjtu/classification/runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








