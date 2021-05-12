'''
该脚本根据训练好的分类模型，遍历detectron测试的xml文件，对xml中object需要重新分类的细胞，取出子图，输入分类模型，输出分类结果，重新写入xml文件中
根据新的xml计算混淆矩阵
'''

import numpy as np
import math
import argparse
import os
import xml.dom.minidom
import codecs 
import cv2
import torch
from confusion_matrix import ConfusionMatrix
from utils import get_network, get_test_dataloader
from conf.global_settings import cell_train_mean, cell_train_std
import torchvision.transforms as transforms
import torch.nn as nn
# 测试集
imgPath = '/data/bone_marrow/data/cell/detection/cases/test-xml-210331/'     # 原图路径
xmlPath = '/home/steadysjtu/classification/xml/'   # 预测结果xml路径
dstPath = '/home/steadysjtu/classification/xml_test_gt/'     # 测试集xml标注
savePath = '/home/steadysjtu/classification/'                # 输出的混淆矩阵路径

cellName = ['原始细胞','早幼粒细胞', '嗜中性-中幼粒细胞','嗜中性-晚幼粒细胞','嗜中性-带形核',
    '嗜中性-分叶核', '嗜酸','嗜碱', '原红细胞', '早幼红细胞','中幼红细胞','晚幼红细胞',
    '成熟淋巴细胞', '单核细胞', '浆细胞','忽略']


def procXml(procPath):
    '''处理xml文件第一行，如果多余则删除'''
    for subdir in os.listdir(procPath):
        #print(subdir)
        xmllist = os.listdir(procPath+subdir)
        for xmlfile in xmllist:
            procfile = procPath+subdir+'/'+xmlfile
            lines = codecs.open(procfile,encoding='utf-8').readlines()
            if lines[0][1] == "?":
                first_line = True
                second_line = True
                for line in lines:
                    if first_line:
                        first_line = False
                    elif second_line:
                        second_line = False
                        codecs.open(procfile, 'w', 'utf-8').writelines(line)
                    else:
                        codecs.open(procfile, 'a', 'utf-8').writelines(line)


def updataXml(imgPath, xmlPath): 
    '''重新二分类，更新xml'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    args = parser.parse_args()

    net = get_network(args)
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(cell_train_mean, cell_train_std)
    ])
    count = 0
    change = 0
    net.load_state_dict(torch.load('/home/steadysjtu/classification/checkpoint/vgg16/38/Wednesday_12_May_2021_13h_00m_05s/vgg16-23-best-0.9005681818181818.pth'))
    net.eval()
    for subdir in os.listdir(xmlPath):
        xmllist = os.listdir(xmlPath+subdir)
        isUpdated = False
        for xmlfile in xmllist:
            srcfile = xmlPath + subdir + "/" + xmlfile              # xml文件名
            image_pre, ext = os.path.splitext(xmlfile)
            imgfile = imgPath + subdir + "/" + image_pre + ".jpg"   # 原图文件名
            img = cv2.imread(imgfile)     # 原图图片
            DOMTree = xml.dom.minidom.parse(srcfile) # 打开xml文档
            collection = DOMTree.documentElement # 得到文档元素对象
            objectlist = collection.getElementsByTagName("object")  # 得到标签名为object的信息
            for objects in objectlist:
                namelist = objects.getElementsByTagName('name')
                name = namelist[0].childNodes[0].data
                if name != '原红细胞' and name != '早幼红细胞': # 对原红细胞和早幼红细胞做二分类
                    continue
                pos0 = cellName.index('原红细胞')
                pos1 = cellName.index('早幼红细胞')
                isUpdated = True         # xml需要更新
                bndbox = objects.getElementsByTagName('bndbox')
                for box in bndbox:
                    x1_list = box.getElementsByTagName('xmin')
                    x1 = int(x1_list[0].childNodes[0].data)
                    y1_list = box.getElementsByTagName('ymin')
                    y1 = int(y1_list[0].childNodes[0].data)
                    x2_list = box.getElementsByTagName('xmax')  # 注意坐标，看是否需要转换
                    x2 = int(x2_list[0].childNodes[0].data)
                    y2_list = box.getElementsByTagName('ymax')
                    y2 = int(y2_list[0].childNodes[0].data)
                try:
                    subimg = img[y1:y2, x1:x2, :]  # 细胞子图
                    subimg = transform_test(subimg)
                    subimg = subimg.unsqueeze(0)
                    # print("sub = ", subimg.shape)
                    output = net(subimg)        ####################################### 输入细胞图片，预测类别
                    # softmax = nn.Softmax()
                    # out = softmax(output)
                    # prob = out.detach().numpy()
                    # with open("/home/steadysjtu/classification/prob.txt", "a") as f:
                    #     f.write(str(prob[0][0])+' '+str(prob[0][1])+'\n')

                    _, preds = output.max(1)
                    preds = preds.item()
                    # print(preds)
                    if preds == 0:
                        preds = pos0
                    else:
                        preds = pos1
                    newname = cellName[preds]      # 根据类别标签输出细胞名称
                    # print(newname)
                    namelist[0].childNodes[0].data = newname
                    count += 1
                    if name != newname:
                        change += 1
                        print("change=", change,',newname = ', newname)
                except:
                    print("error")
            if isUpdated:
                writeXml(DOMTree, srcfile) # 更新xml
    print("count = ", count)
    return 0

def writeXml(DOMTree, xmlfile): # 写xml文件
    with open(xmlfile, 'w', encoding='utf-8') as f:
        DOMTree.writexml(f, addindent=' ', encoding="utf-8")
    lines = codecs.open(xmlfile,encoding='utf-8').readlines()
    first_line = True
    for line in lines:
        if first_line:
            codecs.open(xmlfile, 'w', 'utf-8').writelines("<annotation>\n")
            first_line = False
        else:
            codecs.open(xmlfile, 'a', 'utf-8').writelines(line)

def iou(p1,p2):
    intersection = max(min(p1[2],p2[2])-max(p1[0],p2[0]),0) * max(min(p1[3],p2[3])-max(p1[1],p2[1]),0)
    union = (p1[3]-p1[1])*(p1[2]-p1[0])+(p2[3]-p2[1])*(p2[2]-p2[0])-intersection
    return intersection*1.0/union

def findIndex(name):
    '''细胞名称对应的序号'''
    if name == '原始细胞':
        return 0
    elif name == '早幼粒细胞':
        return 1
    elif name == '嗜中性-中幼粒细胞':
        return 2
    elif name == '嗜中性-晚幼粒细胞':
        return 3
    elif name == '嗜中性-带形核':
        return 4
    elif name == '嗜中性-分叶核':
        return 5
    elif name == '嗜酸':
        return 6
    elif name == '嗜碱':
        return 7
    elif name == '原红细胞':
        return 8
    elif name == '早幼红细胞':
        return 9
    elif name == '中幼红细胞':
        return 10
    elif name == '晚幼红细胞':
        return 11
    elif name == '成熟淋巴细胞':
        return 12
    elif name == '单核细胞':
        return 13
    elif name == '浆细胞':
        return 14
    elif name == '忽略':
        return 15
    return 15

def compareXml(testPath, dstPath):
    '''比较测试集预测xml和标注的xml，求出混淆矩阵'''
    table = np.zeros((17,17))
    nums = np.zeros((16,), dtype=np.int16)
    for subdir in os.listdir(testPath):
        xmllist_test = os.listdir(testPath+subdir)
        for xmlfile in xmllist_test:
            testfile = testPath + subdir + "/" + xmlfile
            dstfile = dstPath + subdir + "/" + xmlfile
            DOMTree_test = xml.dom.minidom.parse(testfile) # 打开xml文档
            DOMTree_dst = xml.dom.minidom.parse(dstfile) 
            collection_test = DOMTree_test.documentElement # 得到文档元素对象
            collection_dst = DOMTree_dst.documentElement 
            objectlist_test = collection_test.getElementsByTagName("object") # 得到标签名为object的信息
            objectlist_dst = collection_dst.getElementsByTagName("object") 
            
            conf_mat = ConfusionMatrix(num_classes=16, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.3)
            my_result = []
            gt_bbox_label = []
            for objects_test in objectlist_test:
                namelist_test = objects_test.getElementsByTagName('name')
                name_test = namelist_test[0].childNodes[0].data
                bndbox_test = objects_test.getElementsByTagName('bndbox')
                p1 = []
                nums[findIndex(name_test)] += 1
                for box in bndbox_test:
                    p_list = box.getElementsByTagName('xmin')
                    p1.append(int(p_list[0].childNodes[0].data))
                    p_list = box.getElementsByTagName('ymin')
                    p1.append(int(p_list[0].childNodes[0].data))
                    p_list = box.getElementsByTagName('xmax')
                    p1.append(int(p_list[0].childNodes[0].data))
                    p_list = box.getElementsByTagName('ymax')
                    p1.append(int(p_list[0].childNodes[0].data))
                p1.append(1)
                p1.append(findIndex(name_test))
                my_result.append(p1)
            for objects_dst in objectlist_dst:
                namelist_dst = objects_dst.getElementsByTagName('name')
                name_dst = namelist_dst[0].childNodes[0].data
                if findIndex(name_dst)==15:
                    continue
                bndbox_dst = objects_dst.getElementsByTagName('bndbox')
                p2 = []
                p2.append(findIndex(name_dst))
                for box in bndbox_dst:
                    p_list = box.getElementsByTagName('xmin')
                    p2.append(int(p_list[0].childNodes[0].data))
                    p_list = box.getElementsByTagName('ymin')
                    p2.append(int(p_list[0].childNodes[0].data))
                    p_list = box.getElementsByTagName('xmax')
                    p2.append(int(p_list[0].childNodes[0].data))
                    p_list = box.getElementsByTagName('ymax')
                    p2.append(int(p_list[0].childNodes[0].data))
                gt_bbox_label.append(p2)
            my_result = np.array(my_result)
            gt_bbox_label = np.array(gt_bbox_label)
            conf_mat.process_batch(my_result, gt_bbox_label)
            tmp = conf_mat.return_matrix()
            table = table + tmp
    np.savetxt(savePath+"table.txt", table, fmt="%d", delimiter="\t") #混淆矩阵
    np.savetxt(savePath+"nums.txt", table, fmt="%d", delimiter="\t")  #预测各类细胞个数
    return 0

if __name__ == '__main__':
    procXml(xmlPath)
    updataXml(imgPath, xmlPath)
    # compareXml(xmlPath, dstPath)
