import os
import xml.etree.ElementTree as ET
import numpy as np
import glob
VOC_BBOX_LABEL_NAMES = (
    "原始细胞",
    "粒细胞系-早幼粒细胞",
    "粒细胞系-嗜中性-中幼粒细胞",
    "粒细胞系-嗜中性-晚幼粒细胞",
    "粒细胞系-嗜中性-杆状核粒细胞",
    "粒细胞系-嗜中性-分叶核粒细胞",
    "粒细胞系-嗜酸性粒细胞",
    "粒细胞系-嗜碱性粒细胞",
    "红细胞系-原红细胞",
    "红细胞系-早幼红细胞",
    "红细胞系-中幼红细胞",
    "红细胞系-晚幼红细胞",
    "淋巴细胞系-成熟淋巴细胞",
    "单核细胞系-成熟单核细胞",
    "浆细胞系-成熟浆细胞",
    # "颗粒型巨核细胞",
    # "产生血小板型巨核细胞",
    # "巨核细胞裸核"
    # '忽略'
    )
# count = 0
label1 = 8
label2 = 9
j = 0
k = 0
pathtrain = '/home/steadysjtu/classification/train/'
pathtest = '/home/steadysjtu/classification/test_gt/'
labelfilestrain = glob.glob(pathname=os.path.join(pathtrain, 'label', '*.txt'))
labelfilestest = glob.glob(pathname=os.path.join(pathtest, 'label', '*.txt'))

# 先清空train_2,test_2
os.system('rm -r /home/steadysjtu/classification/train_2/label/')
os.system('rm -r /home/steadysjtu/classification/train_2/image/')
os.system('rm -r /home/steadysjtu/classification/test_2/label/')
os.system('rm -r /home/steadysjtu/classification/test_2/image/')
os.system('mkdir /home/steadysjtu/classification/train_2/label')
os.system('mkdir /home/steadysjtu/classification/train_2/image')
os.system('mkdir /home/steadysjtu/classification/test_2/label')
os.system('mkdir /home/steadysjtu/classification/test_2/image')


for i in range(len(labelfilestrain)):
    f = open(os.path.join(pathtrain, '/home/steadysjtu/classification/train/label/'+str(i) + '.txt'))
    content = f.readline().strip()
    # print(content.shape)
    if int(content) == label1 or int(content) == label2:
        # print(i)
        os.system('cp /home/steadysjtu/classification/train/label/'+str(i)+'.txt'+' ' + '/home/steadysjtu/classification/train_2/label/'+str(j)+'.txt')
        os.system('cp /home/steadysjtu/classification/train/image/'+str(i)+'.jpg'+' ' + '/home/steadysjtu/classification/train_2/image/'+str(j)+'.jpg')
        # count += 1
        j += 1
        f.close()

for i in range(len(labelfilestest)):
    f = open(os.path.join('/home/steadysjtu/classification/test_gt/label/'+str(i) + '.txt'))
    content = f.readline().strip()
    # print(content.shape)
    if int(content) == label1 or int(content) == label2:
        os.system('cp /home/steadysjtu/classification/test_gt/label/' + str(
            i) + '.txt' + ' ' + '/home/steadysjtu/classification/test_2/label/' + str(k) + '.txt')
        os.system('cp /home/steadysjtu/classification/test_gt/image/' + str(
            i) + '.jpg' + ' ' + '/home/steadysjtu/classification/test_2/image/' + str(k) + '.jpg')
        # count += 1
        k += 1
        f.close()

print("count1 = ", j)
print("count2 = ", k)
