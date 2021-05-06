""" train and test dataset

author baiyu
"""
import os
import sys
import pickle
import glob
import cv2
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset

class CellTrain(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        # if transform is given, we transform data using
        # with open(os.path.join(path, 'train'), 'rb') as cellset:
        #     self.data = pickle.load(cellset, encoding='bytes')
        self.transform = transform
        self.imgfiles = glob.glob(pathname=os.path.join(path, 'image', '*.jpg'))
        self.labelfiles = glob.glob(pathname=os.path.join(path, 'label', '*.txt'))
        self.path = path

    def __len__(self):
        # print("here = ", len(self.labelfiles))
        return len(self.labelfiles)

    def __getitem__(self, ii):
        labelfile = open(os.path.join(self.path, 'label', str(ii)+'.txt'))
        label = int(labelfile.read().strip())
        imgfile = os.path.join(self.path, 'image', str(ii)+'.jpg')
        image = Image.open(imgfile)
        # print("here=", image.shape)
        if self.transform:
            image = self.transform(image)
        # print("image size = ", image.shape)  # (3,32,32)
        # print("label= ", label)  # int
        return label, image

class CellTest(Dataset):
    """cifar100 test dataset, derived from
    torch.utils.data.DataSet
    """

    def __init__(self, path, transform=None):
        # if transform is given, we transform data using
        # with open(os.path.join(path, 'train'), 'rb') as cellset:
        #     self.data = pickle.load(cellset, encoding='bytes')
        self.transform = transform
        self.imgfiles = glob.glob(pathname=os.path.join(path, 'image', '*.jpg'))
        self.labelfiles = glob.glob(pathname=os.path.join(path, 'label', '*.txt'))
        self.path = path

    def __len__(self):
        return len(self.labelfiles)

    def __getitem__(self, ii):
        labelfile = open(os.path.join(self.path, 'label', str(ii) + '.txt'))
        label = int(labelfile.read().strip())
        imgfile = os.path.join(self.path, 'image', str(ii) + '.jpg')
        image = Image.open(imgfile)

        # print("shape = ", image[0][1][2])
        # print(image)
        if self.transform:
            image = self.transform(image)
        # print("image  = ", image)  # (3,32,32)
        # print("label= ", label)  # int
        return label, image

