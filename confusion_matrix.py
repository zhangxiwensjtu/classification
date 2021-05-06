import numpy as np


def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min = 0, a_max = None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(self, num_classes, CONF_THRESHOLD = 0.3, IOU_THRESHOLD = 0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
    
    def process_batch(self, detections, labels):
        '''
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class 测试值
            labels (Array[M, 5]), class, x1, y1, x2, y2  真实值
        Returns:
            None, updates confusion matrix accordingly
        '''
        detections = detections[detections[:, 4] > self.CONF_THRESHOLD]  # 判断每一个数据的第四维，即conf，是否大于阈值，只保留大于的
        gt_classes = labels[:, 0].astype(np.int16)  # 真实值的标签放在这个变量里
        detection_classes = detections[:, 5].astype(np.int16)  # 过滤后的测试标签放在这个变量里

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])  # 真实值和测试值的bbox求iou
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)
        # print("iou=", all_ious)
        # print("index=", want_idx)  # 返回的是一组行列坐标（x1,y1）（x2,y2)......
        all_matches = []
        for i in range(want_idx[0].shape[0]):  # 从行中取值，即按照真实值取值
            all_matches.append([want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]])
            # 这里是把iou中的横纵索引，以及对应的iou存在all_matches中
        all_matches = np.array(all_matches)
        # print(all_matches.shape)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]  # 按照iou的倒序排序

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index = True)[1]]

        for i, label in enumerate(labels):
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                gt_class = gt_classes[i]
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[gt_class, detection_class] += 1
            else:
                gt_class = gt_classes[i]
                self.matrix[gt_class, self.num_classes] += 1  # FN
        
        for i, detection in enumerate(detections):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.matrix[self.num_classes, detection_class] += 1
        # print("FP0=", self.matrix[self.num_classes, 0])
        # print("FP1=", self.matrix[self.num_classes, 1])


    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

