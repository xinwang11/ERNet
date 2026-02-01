import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pvtv2 import pvt_v2_b2
import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Softmax, Dropout

from typing import List, Callable
from torch import Tensor


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class CBR(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttentionwithresidual(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionwithresidual, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output * x


class SpatialAttentionwithresidual(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionwithresidual, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output * x

def weight_init(module):
    for n, m in module.named_children():
        if 'backbone' in n:
            continue
        print('Initialize:' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if "nonlocal_block" in n:
                continue
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Upsample, nn.AdaptiveAvgPool2d, nn.Sigmoid, nn.MaxPool2d, nn.Softmax)):
            pass
        else:
            m.initialize()

def compute_similarity(input_tensor, k=3, dilation=1, sim='cos'):
    B, C, H, W = input_tensor.shape
    unfold_tensor = F.unfold(input_tensor, k, padding=(k // 2) * dilation, dilation=dilation)  # B, CxKxK, HW
    unfold_tensor = unfold_tensor.reshape(B, C, k ** 2, H, W)
    if sim == 'cos':
        similarity = F.cosine_similarity(unfold_tensor[:, :, k * k // 2:k * k // 2 + 1], unfold_tensor[:, :, :], dim=1)
    elif sim == 'dot':
        similarity = unfold_tensor[:, :, k * k // 2:k * k // 2 + 1] * unfold_tensor[:, :, :]
        similarity = similarity.sum(dim=1)
    else:
        raise NotImplementedError

    similarity = torch.cat((similarity[:, :k * k // 2], similarity[:, k * k // 2 + 1:]), dim=1)
    similarity = similarity.view(B, k * k - 1, H, W)
    return similarity

# SEPM
class edgeperception(nn.Module):
    def __init__(self, large_k=9, small_k=5, dilation=1, sim_type='cos'):
        super(edgeperception, self).__init__()
        self.large_k = large_k
        self.small_k = small_k
        self.dilation = dilation
        self.sim_type = sim_type
        self.conv1 = CBR(128, 24, 1)
        self.edge_threshold1 = nn.Parameter(torch.tensor(0.3))
        self.edge_threshold2 = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        large_similarity = compute_similarity(x, k=self.large_k, dilation=self.dilation, sim=self.sim_type)
        large_similarity_mask = (large_similarity < self.edge_threshold1).to(torch.float32)
        large_similarity_mask = torch.norm(large_similarity_mask, p=2, dim=1, keepdim=True)
        x_mask = large_similarity_mask * x + x
        small_similarity = compute_similarity(x_mask, k=self.small_k, dilation=self.dilation, sim=self.sim_type)
        small_similarity_mask = (small_similarity < self.edge_threshold2).to(torch.float32)
        small_similarity_mask = torch.norm(small_similarity_mask, p=2, dim=1, keepdim=True)
        edge_map = x * small_similarity_mask + x

        return edge_map

class EdgeGuidance(nn.Module):
    def __init__(self):
        super(EdgeGuidance, self).__init__()
        self.conv0 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(128)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.GAP1 = nn.AdaptiveAvgPool2d(1)
        self.GMP1 = nn.AdaptiveMaxPool2d(1)
        self.GAP2 = nn.AdaptiveAvgPool2d(1)
        self.GMP2 = nn.AdaptiveMaxPool2d(1)
        self.GAP3 = nn.AdaptiveAvgPool2d(1)
        self.GMP3 = nn.AdaptiveMaxPool2d(1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(128)

        self.ep1 = edgeperception()
        self.ep2 = edgeperception()
        self.ep3 = edgeperception()

    def forward(self, input1):
        out0 = F.relu(self.bn0(self.conv0(input1[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out0_cat = F.relu(self.bn1(self.conv1(torch.cat((input1[1], out0), dim=1))), inplace=True)
        input1[1] = self.GMP1(F.relu(self.bn4(self.conv4(input1[1])), inplace=True))
        out0 = self.GAP1(F.relu(self.bn5(self.conv5(out0)), inplace=True))
        out0 = self.ep1(out0_cat) * input1[1] + out0
        out1 = F.interpolate(out0, size=input1[2].size()[2:], mode='bilinear')
        out1_cat = F.relu(self.bn2(self.conv2(torch.cat((input1[2], out1), dim=1))), inplace=True)
        input1[2] = self.GMP2(F.relu(self.bn6(self.conv6(input1[2])), inplace=True))
        out1 = self.GAP2(F.relu(self.bn7(self.conv7(out1)), inplace=True))
        out1 = self.ep2(out1_cat) * input1[2] + out1
        out2 = F.interpolate(out1, size=input1[3].size()[2:], mode='bilinear')
        out2_cat = F.relu(self.bn3(self.conv3(torch.cat((input1[3], out2), dim=1))), inplace=True)
        input1[3] = self.GMP3(F.relu(self.bn8(self.conv8(input1[3])), inplace=True))
        out2 = self.GAP3(F.relu(self.bn9(self.conv9(out2)), inplace=True))
        out2 = self.ep3(out2_cat) * input1[3] + out2
        return out2

    def initialize(self):
        weight_init(self)

def expend_as(tensor, rep):
    return tensor.repeat(1, rep, 1, 1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Fblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv1_1 = self.ca(conv1) * conv1
        conv1_1 = self.sa(conv1_1) * conv1_1
        conv2 = self.conv2(x)
        x0, x1 = conv2.chunk(2, dim=2)
        x0 = x0.chunk(2, dim=3)
        x1 = x1.chunk(2, dim=3)
        x0 = [self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        x0 = [self.sa(x0[-2]) * x0[-2], self.sa(x0[-1]) * x0[-1]]

        x1 = [self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        x1 = [self.sa(x1[-2]) * x1[-2], self.sa(x1[-1]) * x1[-1]]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x3 = torch.cat((x0, x1), dim=2)

        combined = torch.cat([conv1, conv2], dim=1)
        pooled = self.global_avg_pool(combined)
        pooled = torch.flatten(pooled, 1)
        sigm = self.fc(pooled)

        a = sigm.view(-1, sigm.size(1), 1, 1)
        a1 = 1 - sigm
        a1 = a1.view(-1, a1.size(1), 1, 1)

        y = conv1_1 * a
        y1 = x3 * a1

        combined = torch.cat([y, y1], dim=1)
        out = self.conv3(combined)

        return out


class Sblock(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super(Sblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=size, padding=(size // 2)),
            nn.BatchNorm2d(out_channels)
        )
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.conv3 = CBR(128, 320, 1)

    def forward(self, x, channel_data):
        conv1 = self.conv1(x)
        spatil_data = self.conv2(conv1)

        spatil_data_1 = self.ca(spatil_data) * spatil_data
        spatil_data_1 = self.sa(spatil_data_1) * spatil_data_1

        x0, x1, x2 = channel_data.chunk(3, dim=2)
        x0 = x0.chunk(3, dim=3)
        x1 = x1.chunk(3, dim=3)
        x2 = x2.chunk(3, dim=3)
        x0 = [self.ca(x0[-3]) * x0[-3], self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        x0 = [self.sa(x0[-3]) * x0[-3], self.sa(x0[-2]) * x0[-2], self.sa(x0[-1]) * x0[-1]]

        x1 = [self.ca(x1[-3]) * x1[-3], self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        x1 = [self.sa(x1[-3]) * x1[-3], self.sa(x1[-2]) * x1[-2], self.sa(x1[-1]) * x1[-1]]

        x2 = [self.ca(x2[-3]) * x2[-3], self.ca(x2[-2]) * x2[-2], self.ca(x2[-1]) * x2[-1]]
        x2 = [self.sa(x2[-3]) * x2[-3], self.sa(x2[-2]) * x2[-2], self.sa(x2[-1]) * x2[-1]]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x2 = torch.cat(x2, dim=3)
        x3 = torch.cat((x0, x1, x2), dim=2)

        data3 = torch.add(channel_data, spatil_data)
        data3 = torch.relu(data3)
        data3 = nn.Conv2d(data3.size(1), 1, kernel_size=1, padding=0).cuda()(data3)
        data3 = torch.sigmoid(data3)

        a = expend_as(data3, spatil_data.size(1))
        y = a * spatil_data_1

        a1 = 1 - data3
        a1 = expend_as(a1, channel_data.size(1))
        y1 = a1 * x3

        combined = torch.cat([y, y1], dim=1)
        final_out = self.final_conv(combined)

        return final_out

class AllAtt(nn.Module):
    def __init__(self, in_channels, out_channels, size=3):
        super(AllAtt, self).__init__()
        self.Fblock = Fblock(in_channels, out_channels)
        self.Sblock = Sblock(out_channels, out_channels, size)

    def forward(self, x3, x4):
        Fdata = self.Fblock(x4)
        final_out = self.Sblock(x3, Fdata)
        return final_out

# HSRM
class SDRM1(nn.Module):
    def __init__(self):
        super(SDRM1, self).__init__()
        self.AllAtt = AllAtt(320, 320)
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.conv1 = CBR(512, 320, 1)
        self.conv2 = CBR(128, 320, 1)

    def forward(self, x3, x4):
        x3 = self.conv1(F.pixel_unshuffle(x3, 2))
        final_out = self.AllAtt(x3, x4)

        return final_out

class SDRM2(nn.Module):
    def __init__(self):
        super(SDRM2, self).__init__()
        self.AllAtt = AllAtt(512, 512)
        self.downsample2 = nn.MaxPool2d(2, stride=2)
        self.conv1 = CBR(1280, 512, 1)
        self.conv2 = CBR(128, 320, 1)

    def forward(self, x3, x4):
        x3 = self.conv1(F.pixel_unshuffle(x3, 2))
        final_out = self.AllAtt(x3, x4)

        return final_out

# DDIM
class WFGFM1(nn.Module):
    def __init__(self):
        super(WFGFM1, self).__init__()
        self.conv1 = CBR(64, 128, 1)
        self.conv2 = CBR(128, 64, 1)
        self.context1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64)
        )

        self.context2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128)
        )

        self.context3 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128)
        )

        self.context4 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128)
        )

        self.context5 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128)
        )
        self.conv8 = CBR(32, 64, 1)
        self.conv9 = CBR(128, 256, 1)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.AMP3 = nn.AdaptiveMaxPool2d(3)
        self.AMP5 = nn.AdaptiveMaxPool2d(5)
        self.AMP7 = nn.AdaptiveMaxPool2d(7)
        self.AMP9 = nn.AdaptiveMaxPool2d(9)
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        channel = 128
        self.query_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.value_conv_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_3 = nn.Conv2d(channel, channel, kernel_size=1)
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.gamma_3 = nn.Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

        self.conv_2 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 1)
                                    )
        self.conv_3 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 1)
                                    )

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False),
                                      nn.Conv2d(channel, channel, 1)
                                      )
        self.downsample2 = nn.MaxPool2d(2, stride=2)

    def forward(self, x1, x2):
        nor_weights = F.softmax(self.weight, dim=0)
        x_branch1 = self.GMP(x1)
        x_branch1 = self.context1(x_branch1)
        x_branch2 = self.AMP3(x2)
        x_branch2 = F.interpolate(self.context2(x_branch2), x2.size()[2:], mode='bilinear', align_corners=True) * nor_weights[0]
        x_branch3 = self.AMP5(x2)
        x_branch3 = F.interpolate(self.context3(x_branch3), x2.size()[2:], mode='bilinear', align_corners=True) * nor_weights[1]
        x_branch4 = self.AMP7(x2)
        x_branch4 = F.interpolate(self.context4(x_branch4), x2.size()[2:], mode='bilinear', align_corners=True) * nor_weights[2]
        x_branch5 = self.AMP9(x2)
        x_branch5 = F.interpolate(self.context5(x_branch5), x2.size()[2:], mode='bilinear', align_corners=True) * nor_weights[3]

        x_branch1 = self.conv1(x_branch1)
        x_branch2_fus = x_branch2 + x_branch1 + x_branch3
        x_branch3_fus = x_branch3 + x_branch2_fus + x_branch4 + x_branch2
        x_branch4_fus = x_branch4 + x_branch3_fus + x_branch5 + x_branch3
        x_branch5_fus = x_branch5 + x_branch4_fus + x_branch4

        m_batchsize, C, height, width = x_branch2_fus.size()
        proj_query = self.query_conv(x_branch3_fus).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x_branch4_fus).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value_2 = self.value_conv_2(x_branch2_fus).view(m_batchsize, -1, width * height)
        proj_value_3 = self.value_conv_3(x_branch5_fus).view(m_batchsize, -1, width * height)

        out_2 = torch.bmm(proj_value_2, attention.permute(0, 2, 1))
        out_2 = out_2.view(m_batchsize, C, height, width)
        out_2 = self.conv_2(self.gamma_2 * out_2 + x_branch2_fus)

        out_3 = torch.bmm(proj_value_3, attention.permute(0, 2, 1))
        out_3 = out_3.view(m_batchsize, C, height, width)
        out_3 = self.conv_3(self.gamma_3 * out_3 + x_branch5_fus)

        x2_out = self.conv_out(out_2 + out_3)
        x2_out = self.conv8(F.pixel_shuffle(x2_out, 2))

        x1_out = self.conv2(x_branch1)
        x1_out = x1_out * x1 + x1
        x_out = x1_out + x2_out

        return x_out

class WFGFM2(nn.Module):
    def __init__(self):
        super(WFGFM2, self).__init__()
        self.conv1 = CBR(128, 320, 1)
        self.conv2 = CBR(320, 128, 1)
        self.context1 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128)
        )

        self.context2 = nn.Sequential(
            nn.Conv2d(320, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 320, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(320)
        )

        self.context3 = nn.Sequential(
            nn.Conv2d(320, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 320, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(320)
        )

        self.context4 = nn.Sequential(
            nn.Conv2d(320, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 320, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(320)
        )

        self.context5 = nn.Sequential(
            nn.Conv2d(320, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 320, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(320)
        )
        self.conv8 = CBR(80, 128, 1)
        self.conv9 = CBR(128, 256, 1)
        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.AMP3 = nn.AdaptiveMaxPool2d(3)
        self.AMP5 = nn.AdaptiveMaxPool2d(5)
        self.AMP7 = nn.AdaptiveMaxPool2d(7)
        self.AMP9 = nn.AdaptiveMaxPool2d(9)
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        channel = 320
        self.query_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.value_conv_2 = nn.Conv2d(channel, channel, kernel_size=1)
        self.value_conv_3 = nn.Conv2d(channel, channel, kernel_size=1)
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.gamma_3 = nn.Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

        self.conv_2 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 1)
                                    )
        self.conv_3 = nn.Sequential(BasicConv2d(channel, channel, 3, padding=1),
                                    nn.ReLU(),
                                    nn.Dropout2d(0.1, False),
                                    nn.Conv2d(channel, channel, 1)
                                    )

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False),
                                      nn.Conv2d(channel, channel, 1)
                                      )
        self.downsample2 = nn.MaxPool2d(2, stride=2)

    def forward(self, x1, x2):
        nor_weights = F.softmax(self.weight, dim=0)
        x_branch1 = self.GMP(x1)
        x_branch1 = self.context1(x_branch1)
        x_branch2 = self.AMP3(x2)
        x_branch2 = F.interpolate(self.context2(x_branch2), x2.size()[2:], mode='bilinear', align_corners=True) * nor_weights[0]
        x_branch3 = self.AMP5(x2)
        x_branch3 = F.interpolate(self.context3(x_branch3), x2.size()[2:], mode='bilinear', align_corners=True) * nor_weights[1]
        x_branch4 = self.AMP7(x2)
        x_branch4 = F.interpolate(self.context4(x_branch4), x2.size()[2:], mode='bilinear', align_corners=True) * nor_weights[2]
        x_branch5 = self.AMP9(x2)
        x_branch5 = F.interpolate(self.context5(x_branch5), x2.size()[2:], mode='bilinear', align_corners=True) * nor_weights[3]

        x_branch1 = self.conv1(x_branch1)
        x_branch2_fus = x_branch2 + x_branch1 + x_branch3
        x_branch3_fus = x_branch3 + x_branch2_fus + x_branch4 + x_branch2
        x_branch4_fus = x_branch4 + x_branch3_fus + x_branch5 + x_branch3
        x_branch5_fus = x_branch5 + x_branch4_fus + x_branch4

        m_batchsize, C, height, width = x_branch2_fus.size()
        proj_query = self.query_conv(x_branch3_fus).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x_branch4_fus).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value_2 = self.value_conv_2(x_branch2_fus).view(m_batchsize, -1, width * height)
        proj_value_3 = self.value_conv_3(x_branch5_fus).view(m_batchsize, -1, width * height)

        out_2 = torch.bmm(proj_value_2, attention.permute(0, 2, 1))
        out_2 = out_2.view(m_batchsize, C, height, width)
        out_2 = self.conv_2(self.gamma_2 * out_2 + x_branch2_fus)

        out_3 = torch.bmm(proj_value_3, attention.permute(0, 2, 1))
        out_3 = out_3.view(m_batchsize, C, height, width)
        out_3 = self.conv_3(self.gamma_3 * out_3 + x_branch5_fus)

        x2_out = self.conv_out(out_2 + out_3)
        x2_out = self.conv8(F.pixel_shuffle(x2_out, 2))

        x1_out = self.conv2(x_branch1)
        x1_out = x1_out * x1 + x1
        x_out = x1_out + x2_out

        return x_out

class MappingModule(nn.Module):
    def __init__(self, out_c):
        super(MappingModule, self).__init__()

        nums = [64, 128, 320, 512]
        self.cv1_3 = nn.Sequential(
            nn.Conv2d(nums[0], out_c, kernel_size=3, stride=1,
                      padding=1, dilation=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cv1_1 = nn.Sequential(
            nn.Conv2d(nums[0], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.cv2_3 = nn.Sequential(
            nn.Conv2d(nums[1], out_c, kernel_size=3, stride=1,
                      padding=3, dilation=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cv2_1 = nn.Sequential(
            nn.Conv2d(nums[1], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.cv3_3 = nn.Sequential(
            nn.Conv2d(nums[2], out_c, kernel_size=3, stride=1,
                      padding=5, dilation=5),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cv3_1 = nn.Sequential(
            nn.Conv2d(nums[2], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

        self.cv4_3 = nn.Sequential(
            nn.Conv2d(nums[3], out_c, kernel_size=3, stride=1,
                      padding=5, dilation=5),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.cv4_1 = nn.Sequential(
            nn.Conv2d(nums[3], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, out2, out3, out4, out5):
        o2_1 = self.cv1_3(out2)
        o2_2 = self.cv1_1(out2)
        o2 = o2_1 + o2_2

        o3_1 = self.cv2_3(out3)
        o3_2 = self.cv2_1(out3)
        o3 = o3_1 + o3_2

        o4_1 = self.cv3_3(out4)
        o4_2 = self.cv3_1(out4)
        o4 = o4_1 + o4_2

        o5_1 = self.cv4_3(out5)
        o5_2 = self.cv4_1(out5)
        o5 = o5_1 + o5_2

        return o2, o3, o4, o5

    def initialize(self):
        weight_init(self)

class ResBlk(nn.Module):
    def __init__(self, channel_in=64, channel_out=64, groups=0):
        super(ResBlk, self).__init__()
        self.conv_in = nn.Conv2d(channel_in, 64, 3, 1, 1)
        self.relu_in = nn.ReLU(inplace=True)
        self.conv_out = nn.Conv2d(64, channel_out, 3, 1, 1)
        self.bn_in = nn.BatchNorm2d(64)
        self.bn_out = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x

    def initialize(self):
        weight_init(self)

class ERNet(nn.Module):
    def __init__(self, channel=32):
        super(ERNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.top_layer = ResBlk(512, 320)
        self.lat_layer5 = nn.Conv2d(128, 512, 1, 1, 0)
        self.dec_layer4 = ResBlk(320, 320)
        self.lat_layer_connect4 = nn.Conv2d(320, 320, 1, 1, 0)
        self.lat_layer4 = nn.Conv2d(128, 320, 1, 1, 0)
        self.dec_layer3 = ResBlk(320, 128)
        self.lat_layer_connect3 = nn.Conv2d(128, 128, 1, 1, 0)
        self.lat_layer3 = nn.Conv2d(128, 128, 1, 1, 0)
        self.dec_layer2 = ResBlk(128, 64)
        self.lat_layer_connect2 = nn.Conv2d(64, 64, 1, 1, 0)
        self.lat_layer2 = nn.Conv2d(128, 64, 1, 1, 0)
        self.dec_layer1 = ResBlk(64, 32)
        self.conv_out1 = nn.Sequential(nn.Conv2d(32, 1, 1, 1, 0))

        self.edge_guidance = EdgeGuidance()
        self.mapc1 = MappingModule(128)
        self.e_conv = nn.Conv2d(128, 1, 3, padding=1)
        self.eb = nn.Conv2d(128, 128, 3, padding=1)

        self.WFGFM1 = WFGFM1()
        self.WFGFM2 = WFGFM2()
        self.SDRM1 = SDRM1()
        self.SDRM2 = SDRM2()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        shape = x.shape[2:]
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0] # 64x88x88
        x2 = pvt[1] # 128x44x44
        x3 = pvt[2] # 320x22x22
        x4 = pvt[3] # 512x11x11

        x1eg, x2eg, x3eg, x4eg = self.mapc1(x1, x2, x3, x4)
        edge_branch = self.edge_guidance([x4eg, x3eg, x2eg, x1eg])
        edge_sup = self.eb(edge_branch) # 128x44x44
        edge = self.e_conv(edge_branch)
        e = F.interpolate(edge, size=shape, mode='bilinear', align_corners=True)

        x1 = self.WFGFM1(x1, x2)
        x2 = self.WFGFM2(x2, x3)
        x3 = self.SDRM1(x2, x3)
        x4= self.SDRM2(x3, x4)

        x4 = x4 + F.interpolate(self.lat_layer5(edge_sup), size=x4.shape[2:], mode='bilinear', align_corners=True)
        p4 = self.top_layer(x4)
        p4 = self.dec_layer4(p4)
        p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        p3 = p4 + self.lat_layer_connect4(x3) + F.interpolate(self.lat_layer4(edge_sup), size=x3.shape[2:], mode='bilinear', align_corners=True)

        p3 = self.dec_layer3(p3)
        p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        p2 = p3 + self.lat_layer_connect3(x2) + F.interpolate(self.lat_layer3(edge_sup), size=x2.shape[2:], mode='bilinear', align_corners=True)

        p2 = self.dec_layer2(p2)
        p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        p1 = p2 + self.lat_layer_connect2(x1) + F.interpolate(self.lat_layer2(edge_sup), size=x1.shape[2:], mode='bilinear', align_corners=True)

        p1 = self.dec_layer1(p1)
        p1 = F.interpolate(p1, size=x.shape[2:], mode='bilinear', align_corners=True)

        prediction = self.conv_out1(p1)

        return prediction, self.sigmoid(prediction), e

    def initialize(self):
        if self.training:
            weight_init(self)
