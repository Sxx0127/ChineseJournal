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
import json

myseed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
os.environ['PYTHONHASHSEED'] = str(myseed)
random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)

epoch_num = 5

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    # transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Client():
    def __init__(self, client_num, args):
        self.client_num = client_num
        self.data_len = []
        self.communication_time = [0] * self.client_num
        self.train_loader = []

        self.lr = 0.1
        self.batch_size = 50
        self.model = m.Net()  # m.ResNet(m.ResidualBlock)
        self.last_model = m.Net()
        train_data = datasets.CIFAR10('../../CIFAR-10', train=True, download=True,
                                    transform=transform_train)
        print(range(len(train_data)))
        for i in range(self.client_num):
            data_len = 500
            print("data len is ", data_len)
            self.data_len.append(data_len)
            start_index = int(len(train_data) / self.client_num) * i
            print("the start index is ", start_index)
            data = []
            for j in range(data_len):
                idx = (start_index + j) % len(train_data)
                data.append(train_data[idx])
            train_loader = torch.utils.data.DataLoader(
                data,
                batch_size=self.batch_size, shuffle=True,
                num_workers=0
            )
            self.train_loader.append(train_loader)
            print(i)

        print(self.model)

        total_trainable_params = 0
        for p in self.model.state_dict():
            # for p in self.model.parameters():
            if p.find("num_batches_tracked") != -1:
                continue
            total_trainable_params += self.model.state_dict()[p].numel()
            # total_trainable_params += p.numel()
        print("total parameter num is ", total_trainable_params)
        self.d = total_trainable_params

        self.s = int(math.log(self.d, 2)) + 1
        self.accu_gra = torch.zeros((client_num, total_trainable_params))

    def get_model_gradient(self, idx, args):
        parameters = torch.tensor([])
        for p in self.model.state_dict():
            # for p in self.model.parameters():
            if p.find("num_batches_tracked") != -1:
                continue
            parameters = torch.cat((parameters, self.model.state_dict()[p].view(-1).detach()))
            # parameters = torch.cat((parameters, p.view(-1).detach()))
        last_parameters = torch.tensor([])
        for p in self.last_model.state_dict():
            # for p in self.last_model.parameters():
            if p.find("num_batches_tracked") != -1:
                continue
            last_parameters = torch.cat((last_parameters, self.last_model.state_dict()[p].view(-1).detach()))
            # last_parameters = torch.cat((last_parameters, p.view(-1).detach()))

        self.accu_gra[idx] = last_parameters - parameters + self.accu_gra[idx]
        if args['fit'] == 1:
            sort_gra, indices = torch.sort(self.accu_gra[idx].abs(), descending=True)

            #  拟合幂律分布
            def func(x, alpha, c):
                return c * np.power(x, alpha)
            x = torch.tensor(range(1, self.d + 1))
            popt, pcov = curve_fit(func, x.numpy(), sort_gra.numpy())
            alpha = popt[0]
            c = popt[1]
            print("the alpha of power law is ", alpha)
            print("the c of power law is ", c)

            fitted_updates = c * torch.pow(x, alpha)
            self.accu_gra[idx][indices] = self.accu_gra[idx][indices].sign() * fitted_updates

        k = int(0.01 * len(self.accu_gra[idx]))
        print("the k is ", k)
        value, indices = torch.topk(self.accu_gra[idx].abs(), k)
        new_gra = torch.zeros(self.accu_gra[idx].shape)
        if args['fit'] == 1:
            new_gra[indices] = self.accu_gra[idx][indices]
        else:
            new_gra[indices] = self.accu_gra[idx][indices].sign() * value.mean()
        self.accu_gra[idx][indices] -= new_gra[indices]
        return new_gra

    def PQ_quan(self, parameters, centroids):
        ans = torch.zeros_like(parameters)
        abs_para = parameters.abs()
        p_max = torch.max(abs_para)
        p_min = torch.min(abs_para)
        interval = (p_max - p_min) / centroids
        left = ((abs_para - p_min) / interval).int() * interval + p_min
        right = left + interval
        probability = (abs_para - left) / (right - left)
        probability[probability < 1e-5] = 0
        seed = torch.rand(parameters.shape)
        ans[seed < probability] = right[seed < probability]
        ans[seed >= probability] = left[seed >= probability]
        return ans * torch.sign(parameters)

    def QSGD_quan(self, parameters, centroids):
        norm = torch.norm(parameters, p=2)
        rang = parameters.abs() / norm
        l = (rang * centroids).int()
        p = 1 - (rang * centroids - l)
        p[p < 1e-5] = 0
        seed = torch.rand(parameters.shape)
        xi = torch.zeros_like(parameters)
        xi[seed < p] = l[seed < p] / centroids
        xi[seed >= p] = (l + 1)[seed >= p] / centroids
        ans = norm * torch.sign(parameters) * xi
        return ans

    def set_model(self, model):
        self.model.load_state_dict(model.state_dict())
        self.last_model.load_state_dict(model.state_dict())
        # torch.save(model, "net_params.pkl")
        # self.model = torch.load("net_params.pkl")
        # self.last_model = torch.load("net_params.pkl")

    def train(self, idx, communication_time):
        self.communication_time[idx] += 1
        print("client no is ", idx, " train num is ", self.communication_time[idx])
        self.model = self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        # lr = 0.25 / (math.sqrt(1 + communication_time / 5))
        print("learning rate is ", self.lr)

        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr)
        for epo in range(epoch_num):
            self.model.train()
            for i, (data, label) in enumerate(self.train_loader[idx]):
                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                pred = self.model(data)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print("Train Epoch: {}, iteration: {}, Loss: {}".format
                          (epo, i, loss.item()))

        self.model = self.model.to('cpu')
