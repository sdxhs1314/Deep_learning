import os
import sys
import json
import time
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
from model import resnet34
from utils import train_and_val,plot_acc,plot_loss
import numpy as np
import os
import argparse


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    if not os.path.exists('./weight'):
        os.makedirs('./weight')

    BATCH_SIZE = 128
    # 图像预处理
    # transform = transforms.Compose([
    #     transforms.Pad(4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomCrop(32),
    #     transforms.ToTensor()])

    data_transform = {
        "train": transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
         "val": transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                    }
    train_dataset = torchvision.datasets.CIFAR10(
        'cifar-10-python', train=True, download=True, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    len_train = len(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(
        'cifar-10-python', train=False, download=True, transform=data_transform["val"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # train_dataset = datasets.ImageFolder("archive/training/training/", transform=data_transform["train"])  # 训练集数据
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    #                                            num_workers=4)  # 加载数据
    # len_train = len(train_dataset)
    #
    # val_dataset = datasets.ImageFolder("archive/validation/validation/", transform=data_transform["val"])  # 测试集数据
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    #                                          num_workers=4)  # 加载数据
    # opt = parser.parse_args()
    len_val = len(val_dataset)

    net = resnet34()
    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    optimizer = optim.Adam(net.parameters(), lr=0.01)  # 设置优化器和学习率
    # optimizer = optim.SGD(net.parameters(), lr=opt.lr,
    #                       momentum=0.9, weight_decay=5e-4)

    # 余弦退火有序调整学习率
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)
    epoch = 50

    history = train_and_val(epoch, net, train_loader, len_train,val_loader, len_val,loss_function, optimizer,device)

    plot_loss(np.arange(0,epoch), history)
    plot_acc(np.arange(0,epoch), history)



