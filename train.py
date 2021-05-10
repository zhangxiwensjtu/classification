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

from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from conf.global_settings import cell_train_mean, cell_train_std
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(epoch):
    start = time.time()
    label1 = 0
    label0 = 0
    print("start epoch", epoch)
    net.train()
    # flag = True
    for batch_index, (labels, images) in enumerate(cell_training_loader):
        # if batch_index > 20:
        #     break
        if labels == 8:
            labels = [0]
            labels = torch.from_numpy(np.array(labels))
            label0 += 1
        elif labels == 9:
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
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * 1 + len(images),
            total_samples=len(cell_training_loader.dataset)
        ))
        if epoch <= args.warm:
            warmup_scheduler.step()
    finish = time.time()
    # print("label0=", label0)
    # print("label1=", label1)
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


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
        if labels == 8:
            labels = [0]
            labels = torch.from_numpy(np.array(labels))
            num_0 += 1
        elif labels == 9:
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

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cell_test_loader.dataset),
        correct / len(cell_test_loader.dataset),
        finish - start
    ))
    print("correct =", correct)
    print("num_test=", num_test)
    print("num_0=", num_0)
    print("num_1 = ", num_1)
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
    for (labels, images) in cell_test_loader:
        if labels == 8:
            labels = [0]
            labels = torch.from_numpy(np.array(labels))
            num_0 += 1
        elif labels == 9:
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

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cell_test_loader.dataset),
        correct / len(cell_test_loader.dataset),
        finish - start
    ))
    print("correct =", correct)
    print("num_test=", num_test)
    print("num_0=", num_0)
    print("num_1 = ", num_1)
    return correct / num_test

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)
    # 训练集
    trainpath = '/home/steadysjtu/classification/train_2/'
    # 测试集
    testpath = '/home/steadysjtu/classification/test_2/'  # 细胞子图路径
    logpath = '/home/steadysjtu/classification/vgg16_new2.txt'
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
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}-{accuracy}.pth')

    best_acc = 0.0

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:  # =1
            train_scheduler.step(epoch)
        train(epoch)
        acc_train = eval_training(epoch)
        acc_test = eval_testing(epoch)
        print("acctrain=", acc_train)
        print("acctest=", acc_test)
        with open(logpath, "a") as f:
            f.write("epoch"+str(epoch)+':')
            f.write("acctrain = "+str(acc_train.item()))
            f.write("acctest = "+str(acc_test.item())+'\n')

            # start to save best performance model after learning rate decay to 0.01
        if best_acc < acc_train.item():
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best', accuracy=acc_train)
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc_train
            continue

        # if not epoch % settings.SAVE_EPOCH:
        #     weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular', accuracy=acc_train)
        #     print('saving weights file to {}'.format(weights_path))
        #     torch.save(net.state_dict(), weights_path)

