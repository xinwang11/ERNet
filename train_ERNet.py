import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pdb, os, argparse
from datetime import datetime

from model.ERNet_models import ERNet
from data import get_loader
from utils import clip_gradient, adjust_lr

import pytorch_iou

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=45, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args()

# build models
model = ERNet()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = './dataset/train_dataset/ORSSD/train/image/'
gt_root = './dataset/train_dataset/ORSSD/train/GT/'
edge_root = './dataset/train_dataset/ORSSD/train/edge/'
#image_root = './dataset/train_dataset/EORSSD/train/image/'
#gt_root = './dataset/train_dataset/EORSSD/train/GT/'
#edge_root = './dataset/train_dataset/EORSSD/train/edge/'
#image_root = './dataset/train_dataset/ors-4199/train/image/'
#gt_root = './dataset/train_dataset/ors-4199/train/GT/'
#edge_root = './dataset/train_dataset/ors-4199/train/edge/'
train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)


CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, edges = pack
        images = Variable(images)
        gts = Variable(gts)
        edges = Variable(edges)
        images = images.cuda()
        gts = gts.cuda()
        edges = edges.cuda()

        sal, sal_sig, edge = model(images)
        losse = CE(edge, edges) + IOU(edge, edges)
        losse = losse * 0.3
        loss1 = CE(sal, gts) + IOU(sal_sig, gts)
        loss1 = loss1 * 1.5
        loss = loss1 + losse

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {}, Loss: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data))


    save_path = 'models/ERNet/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'ERNet.pth' + '.%d' % epoch, _use_new_zipfile_serialization=False)

for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
