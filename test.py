#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""
import glob
import os
import torch.nn as nn
import numpy as np
import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader
from conf.global_settings import cell_train_mean, cell_train_std
import cv2

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(cell_train_mean, cell_train_std)
])
if __name__ == '__main__':
    r = 0
    table = np.zeros((103, 3), dtype=np.float32)
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='/home/steadysjtu/classification/checkpoint/vgg16/38/Wednesday_12_May_2021_13h_00m_05s/vgg16-23-best-0.9005681818181818.pth')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)
    testpath = '/home/steadysjtu/classification/test_3/image/'  # 细胞子图路径
    net.load_state_dict(torch.load(args.weights))
    # print(net)
    net.eval()
    files = glob.glob(os.path.join(testpath, '*.jpg'))
    for file in files:
        image = cv2.imread(file)
        subimg = transform_test(image)
        subimg = subimg.unsqueeze(0)
        output = net(subimg)

        softmax = nn.Softmax(dim=1)
        out = softmax(output)
        prob = out.detach().numpy()
        table[r][0] = r
        table[r][1] = prob[0][0]
        table[r][2] = prob[0][1]
        r += 1
    print("save probability of each classes")
    np.savetxt("/home/steadysjtu/classification/prob.txt", table, fmt="%g", delimiter="\t") #混淆矩阵

