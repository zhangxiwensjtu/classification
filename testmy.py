import os
import sys
import argparse
import time
import numpy as np
import glob
import torch
import cv2
from utils import compute_mean_std

if __name__ == '__main__':
    path = '/home/steadysjtu/classification/train/'
    imgfiles = glob.glob(pathname=os.path.join(path, 'image', '*.jpg'))
    i = 0
    mean = [0]*3
    std = [0]*3
    mean = np.array(mean).astype(np.float64)
    std = np.array(std).astype(np.float64)
    for image in imgfiles:
        img = cv2.imread(image)
        i += 1
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        r_std = np.std(r)
        g_std = np.std(g)
        b_std = np.std(b)
        mean += np.array([r_mean, g_mean, b_mean])
        std += np.array([r_std, g_std, b_std])
    print(i)
    mean = np.array(mean)/i
    print("mean=", mean)
    std = np.array(std)/i
    print("std=", std)

