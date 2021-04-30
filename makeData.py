'''该脚本根据xml文件，将骨髓细胞中的细胞子图取出，将图片和标签保存，建立细胞分类数据集'''

import numpy as np
import math
import os
import xml.dom.minidom
import codecs 
import cv2


# 训练集
imgPath1 = '/data/bone_marrow/data/cell/detection/cases/train/'         # 原图路径
xmlPath1 = '/home/xuhang/project/bone_marrow/classification/xml_train/' # xml路径
dstPath1 = '/home/xuhang/project/bone_marrow/classification/train/'     # 子图输出路径

# 测试集
imgPath2 = '/data/bone_marrow/data/cell/detection/cases/test-xml-210331/' # 原图路径
xmlPath2 = '/home/xuhang/project/bone_marrow/classification/xml_test_gt/'   # xml路径
dstPath2 = '/home/xuhang/project/bone_marrow/classification/test_gt/'       # 子图输出路径

'''细胞名称16类，标签0~15'''
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

def makedir(folder_path): # 判断是否存在文件夹如果不存在则创建为文件夹
    if not os.path.exists(folder_path):  
        os.makedirs(folder_path)

def make_cell_image(imgPath, xmlPath, dstPath): 
    '''建立细胞子图数据集'''
    cnt = 0     # 细胞序号
    makedir(dstPath+'image/')
    makedir(dstPath+'label/')
    for subdir in os.listdir(xmlPath):
        xmllist = os.listdir(xmlPath+subdir)
        for xmlfile in xmllist:
            srcfile = xmlPath + subdir + "/" + xmlfile              # xml文件名
            image_pre, ext = os.path.splitext(xmlfile)
            imgfile = imgPath + subdir + "/" + image_pre + ".jpg"   # 原图文件名
            img = cv2.imread(imgfile)     # 原图图片
            DOMTree = xml.dom.minidom.parse(srcfile) # 打开xml文档
            collection = DOMTree.documentElement # 得到文档元素对象
            objectlist = collection.getElementsByTagName("object") # 得到标签名为object的信息
            for objects in objectlist:
                namelist = objects.getElementsByTagName('name')
                name = namelist[0].childNodes[0].data
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
                    label = [findIndex(name)]  # 细胞标签
                    # 保存图片和标签
                    cv2.imencode('.jpg', subimg)[1].tofile(dstPath + 'image/' + '%s.jpg' % cnt)
                    np.savetxt(dstPath+"label/" + '%s.txt' % cnt, label,fmt="%d", delimiter="\t")
                    cnt += 1
                except:
                    continue
    
    return 0

if __name__ == '__main__':
    make_cell_image(imgPath1, xmlPath1, dstPath1) # 创建训练集
    make_cell_image(imgPath2, xmlPath2, dstPath2) # 创建测试集