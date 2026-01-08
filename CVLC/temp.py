from torchvision import datasets, transforms, models
import torch
from torch import nn
import Model as m
import numpy as np
import time
import heapq
import math
import random
import os
import scipy.stats as st
from scipy.optimize import curve_fit

d = 11188362
packet_num = 25
packet_size = 3000 * 8
s = int(math.log(d, 2)) + 1
best_loss = float('inf')


def get_parameterSize(k):
    b = 1
    s = int(math.log(d, 2)) + 1
    for b in range(1, 33):
        if k * (s + b) + 32 * (2 ** b) > packet_size:
            break
    return b - 1


sort_gra = torch.from_numpy(np.loadtxt('sort_gra.txt'))
k = 13850
ks = [int(k / packet_num)] * packet_num  # 初始情况下每个包均分参数
b = get_parameterSize(ks[0])
bs = [b] * packet_num  # 计算此时每个包中参数的大小
print("the parameter number in the packet is {}, the bit in the packet is {}".format(ks[0], bs[0]))
print("the bs is ", bs)

spar_loss = torch.sum(sort_gra[sum(ks):])
print("the k is {}, the spar_loss is {}".format(sum(ks), spar_loss))
quan_loss = ks[0] / (2 * ((2 ** bs[0] - 1) ** 2)) * torch.sum(sort_gra[:sum(ks)])
loss = spar_loss + quan_loss
print("the initial loss is ", loss)

for _ in range(20):
    kL = 0
    for i in range(packet_num - 1):
        kR = kL + ks[i] + ks[i + 1]

        x, y = ks[i], ks[i + 1]
        xb, yb = bs[i], bs[i + 1]
        fx_loss = (x / (2 * ((2 ** xb - 1) ** 2)) * torch.sum(sort_gra[kL: kL + x])) + (
                y / (2 * ((2 ** yb - 1) ** 2)) * torch.sum(sort_gra[kL + x:kR]))

        for xb in range(bs[i] + 1, 33):
            x = int((packet_size - 32 * (2 ** xb)) / (s + xb))
            y = kR - kL - x
            yb = get_parameterSize(y)
            if yb == 0:
                break
            tmp_loss = (x / (2 * ((2 ** xb - 1) ** 2)) * torch.sum(sort_gra[kL:kL + x])) + (
                    y / (2 * ((2 ** yb - 1) ** 2)) * torch.sum(sort_gra[kL + x:kR]))
            if tmp_loss < fx_loss:
                fx_loss = tmp_loss
                ks[i], ks[i + 1] = x, y
                bs[i], bs[i + 1] = xb, yb
        kL += ks[i]

    quan_loss = 0
    accumu_k = 0
    for i in range(packet_num):
        quan_loss += (ks[i] / (2 * ((2 ** bs[i] - 1) ** 2)) * torch.sum(sort_gra[accumu_k:accumu_k + ks[i]]))
        accumu_k += ks[i]
    loss = spar_loss + quan_loss
    print("the compressed loss after optimizer is ", loss)
    if loss < best_loss:
        best_k = sum(ks)
        best_ks = ks
        best_bs = bs
        best_loss = loss
print("the best loss is {} when k is {}".format(best_loss, best_k))
print("the best bs is ", best_bs)
print("the best ks is ", best_ks)


bs = [8, 8, 8, 7, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 7, 7]
ks = [494, 494, 494, 614, 614, 494, 614, 494, 614, 494, 614, 494, 614, 494, 614, 494, 614, 494, 614, 494, 614, 494, 614, 614, 554]
quan_loss = 0
accumu_k = 0
for i in range(packet_num):
    quan_loss += (ks[i] / (2 * ((2 ** bs[i] - 1) ** 2)) * torch.sum(sort_gra[accumu_k:accumu_k + ks[i]]))
    accumu_k += ks[i]
loss = spar_loss + quan_loss
print("the modified loss is ", loss)