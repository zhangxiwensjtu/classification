'''该脚本处理xml文件，二分类的xml结果更新到完整xml中，并计算混淆矩阵'''

import numpy as np
import math
import os
import xml.dom.minidom
import codecs 
from confusion_matrix import ConfusionMatrix

# 测试集真实标注
dstPath = '/home/steadysjtu/classification/xml_test_gt/'
# 测试集预测结果
testPath1 = '/home/steadysjtu/classification/test-20210509/'
# 二分类预测结果
testPath2 = '/home/steadysjtu/classification/xml/'
# 混淆矩阵保存路径
savePath = '/home/steadysjtu/classification/'

def makedir(folder_path): # 判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(folder_path):  
        os.makedirs(folder_path)

def iou(p1,p2):
    intersection = max(min(p1[2],p2[2])-max(p1[0],p2[0]),0) * max(min(p1[3],p2[3])-max(p1[1],p2[1]),0)
    union = (p1[3]-p1[1])*(p1[2]-p1[0])+(p2[3]-p2[1])*(p2[2]-p2[0])-intersection
    return intersection*1.0/union

def findIndex(name):
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

def compareXml(testPath, dstPath):
    '''计算混淆矩阵'''
    table = np.zeros((17,17), dtype=np.int16)
    for subdir in os.listdir(testPath):  # 遍历第一阶段生成的子文件夹
        xmllist_test = os.listdir(testPath+subdir)
        for xmlfile in xmllist_test:  # 遍历第一阶段生成的xml文件
            testfile = testPath + subdir + "/" + xmlfile
            dstfile = dstPath + subdir + "/" + xmlfile
            DOMTree_test = xml.dom.minidom.parse(testfile) # 打开xml文档
            DOMTree_dst = xml.dom.minidom.parse(dstfile) 
            collection_test = DOMTree_test.documentElement # 得到文档元素对象
            collection_dst = DOMTree_dst.documentElement 
            objectlist_test = collection_test.getElementsByTagName("object") # 得到标签名为object的信息
            objectlist_dst = collection_dst.getElementsByTagName("object") 

            for objects_test in objectlist_test:
                namelist_test = objects_test.getElementsByTagName('name')
                name_test = namelist_test[0].childNodes[0].data
                bndbox_test = objects_test.getElementsByTagName('bndbox')
                p1 = []
                isfound = False
                for box in bndbox_test:
                    p_list = box.getElementsByTagName('xmin')
                    p1.append(int(p_list[0].childNodes[0].data))
                    p_list = box.getElementsByTagName('ymin')
                    p1.append(int(p_list[0].childNodes[0].data))
                    p_list = box.getElementsByTagName('xmax')
                    p1.append(int(p_list[0].childNodes[0].data))
                    p_list = box.getElementsByTagName('ymax')
                    p1.append(int(p_list[0].childNodes[0].data))
                for objects_dst in objectlist_dst:
                    namelist_dst = objects_dst.getElementsByTagName('name')
                    name_dst = namelist_dst[0].childNodes[0].data
                    bndbox_dst = objects_dst.getElementsByTagName('bndbox')
                    p2 = []
                    for box in bndbox_dst:
                        p_list = box.getElementsByTagName('xmin')
                        p2.append(int(p_list[0].childNodes[0].data))
                        p_list = box.getElementsByTagName('ymin')
                        p2.append(int(p_list[0].childNodes[0].data))
                        p_list = box.getElementsByTagName('xmax')
                        p2.append(int(p_list[0].childNodes[0].data))
                        p_list = box.getElementsByTagName('ymax')
                        p2.append(int(p_list[0].childNodes[0].data))
                    if iou(p1, p2) > 0.5:
                        isfound = True
                        table[findIndex(name_dst)][findIndex(name_test)] += 1
                if isfound == False:
                    table[16][findIndex(name_test)] += 1
    
    np.savetxt(savePath+"/table.txt", table, fmt="%d", delimiter="\t") #混淆矩阵
    return 0

def reClassify(testPath, dstPath):
    '''再分类后的xml同步到完整xml中
    testPath是完整xml路径，dstPath是二分类xml路径
    '''
    count = 0
    cou = 0
    for subdir in os.listdir(dstPath):
        print(subdir)
        xmllist = os.listdir(dstPath+subdir)
        for xmlfile in xmllist:
            isUpdated = False
            dstfile = dstPath + subdir + "/" + xmlfile
            testfile = testPath + subdir + "/" + xmlfile
            DOMTree_dst = xml.dom.minidom.parse(dstfile) # 打开xml文档
            DOMTree_test = xml.dom.minidom.parse(testfile) 
            collection_dst = DOMTree_dst.documentElement # 得到文档元素对象
            collection_test = DOMTree_test.documentElement 
            objectlist_dst = collection_dst.getElementsByTagName("object") # 得到标签名为object的信息
            objectlist_test = collection_test.getElementsByTagName("object")
            for objects_dst in objectlist_dst:
                namelist_dst = objects_dst.getElementsByTagName('name')
                name_dst = namelist_dst[0].childNodes[0].data
                bndbox_dst = objects_dst.getElementsByTagName('bndbox')
                for box in bndbox_dst:
                    x1_list = box.getElementsByTagName('xmin')
                    x1_dst = int(x1_list[0].childNodes[0].data)
                    y1_list = box.getElementsByTagName('ymin')
                    y1_dst = int(y1_list[0].childNodes[0].data)
                    x2_list = box.getElementsByTagName('xmax')  # 注意坐标，看是否需要转换
                    x2_dst = int(x2_list[0].childNodes[0].data)
                    y2_list = box.getElementsByTagName('ymax')
                    y2_dst = int(y2_list[0].childNodes[0].data)
                for objects_test in objectlist_test:
                    namelist_test = objects_test.getElementsByTagName('name')
                    name_test = namelist_test[0].childNodes[0].data
                    if name_test == '嗜中性-分叶核' or name_test == '嗜中性-带形核':
                        bndbox_test = objects_test.getElementsByTagName('bndbox')
                        for box in bndbox_test:
                            x1_list = box.getElementsByTagName('xmin')
                            x1_test = int(x1_list[0].childNodes[0].data)
                            y1_list = box.getElementsByTagName('ymin')
                            y1_test = int(y1_list[0].childNodes[0].data)
                            x2_list = box.getElementsByTagName('xmax')  # 注意坐标，看是否需要转换
                            x2_test = int(x2_list[0].childNodes[0].data)
                            y2_list = box.getElementsByTagName('ymax')
                            y2_test = int(y2_list[0].childNodes[0].data)
                        if abs(x1_dst-x1_test)<10 and abs(y1_dst-y1_test)<10 and abs(x2_dst-x2_test)<10 and abs(y2_dst-y2_test)<10:
                            isUpdated = True
                            cou += 1
                            print("cou=", cou)
                            namelist_test[0].childNodes[0].data = namelist_dst[0].childNodes[0].data
            if isUpdated:
                with open(testfile, 'w', encoding='utf-8') as f:
                    DOMTree_test.writexml(f, addindent=' ', encoding="utf-8")
                    # count += 1
                    # print(count)
                lines = codecs.open(testfile,encoding='utf-8').readlines()
                first_line = True
                for line in lines:
                    if first_line:
                        codecs.open(testfile, 'w', 'utf-8').writelines("<annotation>\n")
                        first_line = False
                    else:
                        codecs.open(testfile, 'a', 'utf-8').writelines(line)
    return 0


def compareXml2(testPath, dstPath, savePath):
    table = np.zeros((17, 17))

    for subdir in os.listdir(testPath):
        xmllist_test = os.listdir(testPath + subdir)
        for xmlfile in xmllist_test:
            testfile = testPath + subdir + "/" + xmlfile
            dstfile = dstPath + subdir + "/" + xmlfile
            DOMTree_test = xml.dom.minidom.parse(testfile)  # 打开xml文档
            DOMTree_dst = xml.dom.minidom.parse(dstfile)
            collection_test = DOMTree_test.documentElement  # 得到文档元素对象
            collection_dst = DOMTree_dst.documentElement
            objectlist_test = collection_test.getElementsByTagName("object")  # 得到标签名为object的信息
            objectlist_dst = collection_dst.getElementsByTagName("object")

            conf_mat = ConfusionMatrix(num_classes=16, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.3)
            my_result = []
            gt_bbox_label = []
            for objects_test in objectlist_test:
                namelist_test = objects_test.getElementsByTagName('name')
                name_test = namelist_test[0].childNodes[0].data
                bndbox_test = objects_test.getElementsByTagName('bndbox')
                p1 = []
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
                if findIndex(name_dst) == 15:
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
            if gt_bbox_label.shape == (0,):
                continue
            # print("my_result = ", my_result)
            # print("gt = ", gt_bbox_label)
            conf_mat.process_batch(my_result, gt_bbox_label)
            tmp = conf_mat.return_matrix()
            table = table + tmp
    np.savetxt(savePath + "table.txt", table, fmt="%d", delimiter="\t")
    return 0


if __name__ == '__main__':
    procXml(testPath2)
    procXml(dstPath)
    reClassify(testPath1, testPath2)  # 更新完整xml
    compareXml(testPath1, dstPath)   # 计算新的混淆矩阵
