from statistic_utils import *

cellName = ['原始细胞',
            '早幼粒细胞',
            '嗜中性-中幼粒细胞',
            '嗜中性-晚幼粒细胞',
            '嗜中性-带形核',
            '嗜中性-分叶核',
            '嗜酸',
            '嗜碱',
            '原红细胞',
            '早幼红细胞',
            '中幼红细胞',
            '晚幼红细胞',
            '成熟淋巴细胞',
            '单核细胞',
            '浆细胞',
            '忽略']

imgPath = '/data/bone_marrow/data/cell/detection/cases/test-xml-210331/'     # 原图路径
first_stage_xml_2class = '/home/steadysjtu/classification/xml/'   # 第一阶段预测结果xml路径
xml_test_gt = '/home/steadysjtu/classification/xml_test_gt/'     # 测试集xml标注
savePath = '/home/steadysjtu/classification/'                # 输出的混淆矩阵路径
test_xml_all = '/home/steadysjtu/classification/test-20210509/'  # 测试集预测结果
weight = '/home/steadysjtu/classification/checkpoint/vgg16/45/Friday_14_May_2021_21h_49m_34s/vgg16-58-best-0.930550284629981.pth'
target_name1 = cellName[4]
target_name2 = cellName[5]
# 处理第一阶段生成的xml的文件格式
print("processing xml file")
procXml(first_stage_xml_2class)
# 第二阶段测试
print("testing again, target cell name is ", target_name1, " and ", target_name2)
two_class_test(imgPath, first_stage_xml_2class, weight_path=weight, target_name1=target_name1, target_name2=target_name2,netname='vgg16')  # 这一步会将测试结果更新到first_stage_xml_2class中
print("two class test result has been stored in ",first_stage_xml_2class)
# 将二分类结果整合到所有xml中
print("renew all the xml file")
renewxml(test_xml_all, first_stage_xml_2class, target_name1, target_name2)
# 计算混淆矩阵
compareXml(test_xml_all, xml_test_gt, savePath='/home/steadysjtu/classification/')


