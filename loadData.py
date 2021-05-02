'''该脚本加载细胞分类数据集'''

import numpy as np
import math
import os
import xml.dom.minidom
import codecs 
import cv2
import torch

# 训练集
imgPathtrain = '/home/steadysjtu/classification/train/image/'  # 细胞子图路径
labelPathtrain = '/home/steadysjtu/classification/train/label/'  # 标签路径

# 测试集
imgPathtest = '/home/steadysjtu/classification/test_gt/image/'   # 细胞子图路径
labelPathtest = '/home/steadysjtu/classification/test_gt/label/'  # 标签路径

if __name__ == '__main__':
    filename = "10.txt"
    pre, ext = os.path.splitext(filename)
    labelfile = labelPathtrain + filename
    imgfile = imgPathtrain + pre + ".jpg"
    img = cv2.imread(imgfile).transpose(2, 0, 1)  # c,h,w
    label = np.array(int(np.loadtxt(labelfile)))
    # print(label)
    # print(img.shape)