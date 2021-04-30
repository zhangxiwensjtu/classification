'''该脚本加载细胞分类数据集'''

import numpy as np
import math
import os
import xml.dom.minidom
import codecs 
import cv2


# 训练集
imgPath1 = '/home/xuhang/project/bone_marrow/classification/train/image/'     # 细胞子图路径
labelPath1 = '/home/xuhang/project/bone_marrow/classification/train/label/' # 标签路径

# 测试集
imgPath2 = '/home/xuhang/project/bone_marrow/classification/test_gt/image/'     # 细胞子图路径
labelPath2 = '/home/xuhang/project/bone_marrow/classification/test_gt/label/' # 标签路径

if __name__ == '__main__':
    filename = "10.txt"
    pre, ext = os.path.splitext(filename)
    labelfile = labelPath1 + filename
    imgfile = imgPath1 + pre + ".jpg"
    img = cv2.imread(imgfile)
    label = int(np.loadtxt(labelfile))
    print(label)
    print(img.shape)