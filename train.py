# coding=utf-8
# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from conf.global_settings import cell_train_mean, cell_train_std
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
def train(epoch):
    start = time.time()
    label1 = 0
    label0 = 0
    print("epoch= ", epoch, end=' ')
    net.train()
    # flag = True
    for batch_index, (labels, images) in enumerate(cell_training_loader):
        # if batch_index > 20:
        #     break
        if labels == typea:
            labels = [0]
            labels = torch.from_numpy(np.array(labels))
            label0 += 1
        elif labels == typeb:
            labels = [1]
            labels = torch.from_numpy(np.array(labels))
            label1 += 1
        else:
            continue
        # if flag == True:
        #     print(images.shape)
        #     img = images.clone()[0][:, :, :].numpy()
        #     img = img*255
        #     img = img.transpose(1, 2, 0)
        #     print(img.shape)
        #     cv2.imwrite('/home/steadysjtu/classification/verify.jpg', img)
        #     flag = False
        #     print("save figure")
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        #     loss.item(),
        #     optimizer.param_groups[0]['lr'],
        #     epoch=epoch,
        #     trained_samples=batch_index * 1 + len(images),
        #     total_samples=len(cell_training_loader.dataset)
        # ))
        if epoch <= args.warm:
            warmup_scheduler.step()
    finish = time.time()
    # print("label0=", label0)
    # print("label1=", label1)
    # print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    num_test = 0
    num_0 = 0
    num_1 = 0
    for (labels, images) in cell_train_test_loader:
        if labels == typea:
            labels = [0]
            labels = torch.from_numpy(np.array(labels))
            num_0 += 1
        elif labels == typeb:
            labels = [1]
            labels = torch.from_numpy(np.array(labels))
            num_1 += 1
        else:
            continue

        num_test += 1
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        # print("images.shape =", images)
        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)

        # print("preds =", preds)
        # print("labels = ", labels)
        # print("output = ", outputs)
        # print("images=", images[0][0][:])
        correct += preds.eq(labels).sum()
    print("trainloss = ", test_loss, end=' ')
    finish = time.time()
    # print('Evaluating Network.....')
    # print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
    #     epoch,
    #     test_loss / len(cell_test_loader.dataset),
    #     correct / len(cell_test_loader.dataset),
    #     finish - start
    # ))
    # print("correct =", correct)
    # print("num_test=", num_test)
    # print("num_0=", num_0)
    # print("num_1 = ", num_1)
    return correct / num_test


@torch.no_grad()
def eval_testing(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    num_test = 0
    num_0 = 0
    num_1 = 0
    TP0 = 0
    TN0 = 0
    FP0 = 0
    FN0 = 0
    for (labels, images) in cell_test_loader:
        if labels == typea:
            labels = [0]
            labels = torch.from_numpy(np.array(labels))
            num_0 += 1
        elif labels == typeb:
            labels = [1]
            labels = torch.from_numpy(np.array(labels))
            num_1 += 1
        else:
            continue

        num_test += 1
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        # print("images.shape =", images)
        outputs = net(images)

        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        if preds == 0 and labels == 0:
            TP0 += 1
        elif preds == 1 and labels == 1:
            TN0 += 1
        elif preds == 0 and labels == 1:
            FP0 += 1
        else:
            FN0 += 1

        # print("preds =", preds)
        # print("labels = ", labels)
        # print("output = ", outputs)
        # print("images=", images[0][0][:])
        correct += preds.eq(labels).sum()
    # acc0 = num_0_co / num_0
    # acc1 = num_1_co / num_1
    print("TP0 = ", TP0, end=' ')
    print("FP0 = ", FP0, end=' ')
    print("FN0 = ", FN0, end=' ')
    print("TN0 = ", TN0, end=' ')
    # print("num_test=", num_test)
    print("testloss= ", test_loss, end=' ')

    finish = time.time()
    # print('Evaluating Network.....')
    # print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
    #     epoch,
    #     test_loss / len(cell_test_loader.dataset),
    #     correct / len(cell_test_loader.dataset),
    #     finish - start
    # ))
    # print("correct =", correct)
    # print("num_test=", num_test)
    # print("num_0=", num_0)
    # print("num_1 = ", num_1)

    return (TP0+TN0)/num_test

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()
    typea = 8
    typeb = 9
    net = get_network(args)
    # 训练集
    trainpath = '/home/steadysjtu/classification/train_2/'
    # 测试集
    testpath = '/home/steadysjtu/classification/test_3/'  # 细胞子图路径

    # data preprocessing:
    # 预处理https://www.cnblogs.com/wanghui-garcia/p/11448460.html
    cell_training_loader = get_training_dataloader(
        path=trainpath,
        mean=cell_train_mean,
        std=cell_train_std,
        num_workers=4,
        batch_size=1,
        shuffle=True
    )
    cell_test_loader = get_test_dataloader(
        path=testpath,
        mean=cell_train_mean,
        std=cell_train_std,
        num_workers=4,
        batch_size=1,
        shuffle=True
    )
    cell_train_test_loader = get_test_dataloader(
        path=trainpath,
        mean=cell_train_mean,
        std=cell_train_std,
        num_workers=4,
        batch_size=1,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    iter_per_epoch = len(cell_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, 10000 * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, '38', settings.TIME_NOW)
    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    logpath = os.path.join(checkpoint_path, 'log.txt')
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}-{accuracy}.pth')

    best_acc = 0.0
    # pretrained_weight = torch.load('/home/steadysjtu/classification/checkpoint/vgg16/Monday_10_May_2021_10h_23m_49s/vgg16-257-best-0.9106575846672058.pth')
    # for key, v in pretrained_weight.items():
    #     print('a=',key, v.size())
    pretrained_weight = model_zoo.load_url(model_urls['vgg16'])

    model_dict = net.state_dict()  # 获取当前网络的键值
    # pretrained_dict = {k: v for k, v in pretrained_weight.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # for key, v in model_dict.items():
    #     print(key, v.size())
    del[pretrained_weight['classifier.0.weight']]
    del[pretrained_weight['classifier.6.weight']]
    del[pretrained_weight['classifier.6.bias']]
    pretrained_dict = {k: v for k, v in pretrained_weight.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # for name, value in net.named_parameters():
    #     if name.find('classifier') == -1:
    #         value.requires_grad = False
    #         # print("1=", name)
    # for name, value in net.named_parameters():
    #     print(name, value.requires_grad)
    # input()
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:  # =1
            train_scheduler.step(epoch)
        train(epoch)
        acc_train = eval_training(epoch)
        # print("acc_train=", acc_train)
        # break
        acc_test = eval_testing(epoch)
        # print("acctrain=", acc_train)
        print("acctest=", acc_test)
        with open(logpath, "a") as f:
            f.write("epoch"+str(epoch)+':')
            f.write("acctrain = "+str(acc_train.item())+' ')
            f.write("acctest = "+str(acc_test)+'\n')

            # start to save best performance model after learning rate decay to 0.01
        if best_acc < acc_test:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best', accuracy=acc_test)
            # print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc_test
            continue
        if epoch == 100:
            torch.load(weights_path)
        # if not epoch % settings.SAVE_EPOCH:
        #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular', accuracy=acc_train)
        #     print('saving weights file to {}'.format(weights_path))
        #     torch.save(net.state_dict(), weights_path)

