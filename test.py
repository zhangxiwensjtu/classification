#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""
import numpy as np
import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader
from conf.global_settings import cell_train_mean, cell_train_std

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='/home/steadysjtu/classification/checkpoint/vgg16/Wednesday_05_May_2021_21h_57m_53s/vgg16-128-best-0.8126984238624573.pth')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)
    testpath = '/home/steadysjtu/classification/test_gt/'  # 细胞子图路径

    cell_test_loader = get_test_dataloader(
        path=testpath,
        mean=cell_train_mean,
        std=cell_train_std,
        num_workers=4,
        batch_size=1,
        shuffle=True
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    label1 = 0
    label0 = 0
    with torch.no_grad():
        for n_iter, (label, image) in enumerate(cell_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cell_test_loader)))
            if label == 8:
                label = [0]
                label = torch.from_numpy(np.array(label))
                label0 += 1
            elif label == 9:
                label = [1]
                label = torch.from_numpy(np.array(label))
                label1 += 1
            else:
                continue
            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')


            output = net(image)
            _, pred = output.topk(2, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            # correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / 348)
    # print("Top 5 err: ", 1 - correct_5 / 348)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
