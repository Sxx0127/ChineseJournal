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
from utils import batch_data, word_to_indices, letter_to_vec

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

torch.set_printoptions(precision=8)

epoch_num = 5

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    # transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

packet_size = 1500 * 8
packet_num = 8
remain_packet_num = 2


class Client():
    def __init__(self, client_num, args, batch_size):
        self.client_num = client_num
        self.data_len = []
        self.communication_time = [0] * self.client_num
        self.train_loader = []

        self.best_k = 0
        self.best_ks = []
        self.best_bs = []
        self.best_loss = float('inf')
        self.unselect_bit = 1
        global packet_num, remain_packet_num

        if args['dataset'] == "CIFAR-10":
            self.max_optim_iter = 10
            self.optim_interval = 10
            self.max_init_iter = 10
            self.lr = 0.1
            self.batch_size = batch_size
            self.model = m.Net()  # m.ResNet(m.ResidualBlock)
            self.last_model = m.Net()  # torch.load("../cifar.pkl")
            train_data = datasets.CIFAR10('~/GraduatedPaper/dataset/CIFAR-10', train=True, download=True,
                                          transform=transform_train)
            print(range(len(train_data)))
            data_len = int(50000 / self.client_num)
            print("data len is ", data_len)
            if args['iid'] == "IID":
                for i in range(self.client_num):
                    self.data_len.append(data_len)
                    start_index = int(len(train_data) / self.client_num) * i
                    print("the start index is ", start_index)
                    data = []
                    for j in range(data_len):
                        idx = (start_index + j) % len(train_data)
                        data.append(train_data[idx])
                    train_loader = torch.utils.data.DataLoader(
                        data,
                        batch_size=self.batch_size, shuffle=True
                    )
                    self.train_loader.append(train_loader)
                    print(i)
            elif args['iid'] == 'NIID':
                self.label_index = [[] for i in range(10)]
                for i, (d, l) in enumerate(train_data):
                    self.label_index[l].append(i)
                start_index = [0] * 10
                choose_label = list(range(10))
                for i in range(self.client_num):
                    print(i)
                    if len(choose_label) == 0:
                        choose_label = list(range(10))
                    for j in range(10):
                        if (start_index[j] > (5000 - int(data_len / 5))) and (j in choose_label):
                            choose_label.remove(j)
                    all_samples = []
                    label = random.sample(choose_label, 5)
                    choose_label.remove(label[0])
                    choose_label.remove(label[1])
                    choose_label.remove(label[2])
                    choose_label.remove(label[3])
                    choose_label.remove(label[4])
                    # if i % 2 == 0:
                    #     label = random.sample(choose_label, 5)
                    # else:
                    #     label = [val for val in choose_label if val not in label]
                    for j in label:
                        all_samples.extend(random.sample
                                           (self.label_index[j][start_index[j]:start_index[j] + int(data_len / 5)],
                                            int(data_len / 5)))
                        start_index[j] += int(data_len / 5)
                    print("the choose label are ", label)
                    print("data_len is ", len(all_samples))
                    self.data_len.append(len(all_samples))
                    data = []
                    for j in all_samples:
                        data.append(train_data[j])
                    train_loader = torch.utils.data.DataLoader(
                        data,
                        batch_size=self.batch_size, shuffle=True,
                        num_workers=0
                    )
                    self.train_loader.append(train_loader)
        elif args['dataset'] == "CIFAR-100":
            self.max_optim_iter = 20
            self.optim_interval = 50
            self.max_init_iter = 10
            packet_num = 57
            remain_packet_num = 33
            self.batch_size = batch_size
            self.lr = 0.1
            self.model = m.CNNCifar100()  # m.ResNet(m.ResidualBlock)
            self.last_model = m.CNNCifar100()
            train_data = datasets.CIFAR100('~/GraduatedPaper/dataset/CIFAR-100', train=True, download=True,
                                           transform=transform_train)
            print(range(len(train_data)))
            data_len = int(50000 / self.client_num)
            print("data len is ", data_len)
            if args['iid'] == "IID":
                for i in range(self.client_num):
                    self.data_len.append(data_len)
                    start_index = int(len(train_data) / self.client_num) * i
                    print("the start index is ", start_index)
                    data = []
                    for j in range(data_len):
                        idx = (start_index + j) % len(train_data)
                        data.append(train_data[idx])
                    train_loader = torch.utils.data.DataLoader(
                        data,
                        batch_size=self.batch_size, shuffle=True
                    )
                    self.train_loader.append(train_loader)
                    print(i)
            elif args['iid'] == 'NIID':
                self.label_index = [[] for i in range(100)]
                for i, (d, l) in enumerate(train_data):
                    self.label_index[l].append(i)
                start_index = [0] * 100
                choose_label = list(range(100))
                for i in range(self.client_num):
                    print(i)
                    if len(choose_label) == 0:
                        choose_label = list(range(100))
                    for j in range(100):
                        if (start_index[j] > (500 - int(data_len / 20))) and (j in choose_label):
                            choose_label.remove(j)

                    all_samples = []
                    label = random.sample(choose_label, 20)
                    for i in range(20):
                        choose_label.remove(label[i])
                    for j in label:
                        all_samples.extend(random.sample
                                           (self.label_index[j][start_index[j]:start_index[j] + int(data_len / 20)],
                                            int(data_len / 20)))
                        start_index[j] += int(data_len / 20)
                    print("the choose label are ", label)
                    print("data_len is ", len(all_samples))
                    self.data_len.append(len(all_samples))
                    data = []
                    for j in all_samples:
                        data.append(train_data[j])
                    train_loader = torch.utils.data.DataLoader(
                        data,
                        batch_size=self.batch_size, shuffle=True,
                        num_workers=0
                    )
                    self.train_loader.append(train_loader)
        elif args['dataset'] == "FEMNIST":
            self.max_optim_iter = 10
            self.optim_interval = 10
            self.max_init_iter = 10
            packet_num = 6
            remain_packet_num = 4
            self.lr = 0.05
            self.batch_size = batch_size
            self.model = m.FEMNISTNet()
            self.last_model = m.FEMNISTNet()
            train_data_dir = os.path.join(r'/home/suxiaoxin/GraduatedPaper/dataset/leaf-master/data/femnist/data/train')
            train_files = os.listdir(train_data_dir)
            cnum = 0
            for f in train_files:
                file_path = os.path.join(train_data_dir, f)
                with open(file_path, 'r') as inf:
                    cdata = json.load(inf)
                for i in range(len(cdata['num_samples'])):
                    self.data_len.append(cdata['num_samples'][i])
                    client = cdata['users'][i]
                    x_train, y_train = cdata['user_data'][client]['x'], cdata['user_data'][client]['y']
                    x_train = torch.Tensor(x_train).type(torch.float32)
                    y_train = torch.Tensor(y_train).type(torch.int64)
                    data = [(x.view(1, 28, 28), y) for x, y in zip(x_train, y_train)]
                    train_loader = torch.utils.data.DataLoader(
                        data,
                        batch_size=self.batch_size, shuffle=True,
                        num_workers=0
                    )
                    self.train_loader.append(train_loader)
                    print("{}th data len is {}".format(i, len(y_train)))
                    cnum += 1
                    if cnum >= self.client_num:
                        break
                if cnum >= self.client_num:
                    print("the reading file is ", f)
                    break
        elif args['dataset'] == "shakespeare":
            packet_num = 20
            self.lr = 0.1
            self.batch_size = 128
            self.model = m.LSTMModel()  # m.ResNet(m.ResidualBlock)
            self.last_model = m.LSTMModel()
            train_data_dir = os.path.join('../../leaf-master/data/shakespeare/data/train')
            train_files = os.listdir(train_data_dir)
            cnum = 0
            for f in train_files:
                file_path = os.path.join(train_data_dir, f)
                with open(file_path, 'r') as inf:
                    cdata = json.load(inf)
                for i in range(len(cdata['num_samples'])):
                    self.data_len.append(cdata['num_samples'][i])
                    client = cdata['users'][i]
                    x_train, y_train = cdata['user_data'][client]['x'], cdata['user_data'][client]['y']
                    x = [word_to_indices(word) for word in x_train]
                    y = [letter_to_vec(c) for c in y_train]
                    x = torch.LongTensor(x)
                    y = torch.LongTensor(y)
                    data = [(w, c) for w, c in zip(x, y)]
                    train_loader = torch.utils.data.DataLoader(
                        data,
                        batch_size=self.batch_size, shuffle=True,
                        num_workers=0
                    )
                    self.train_loader.append(train_loader)
                    print("{}th data len is {}".format(i, len(y_train)))
                    cnum += 1
                    if cnum >= self.client_num:
                        break
                if cnum >= self.client_num:
                    print("the reading file is ", f)
                    break

        print(self.model)

        
        # total_trainable_params = sum(self.model.state_dict()[p].numel() for p in self.model.state_dict() if p.find("num_batches_tracked") == -1)
        total_trainable_params = sum(p.numel() for p in self.model.parameters())
        print("total parameter num is ", total_trainable_params)
        self.d = total_trainable_params

        s = int(math.log(self.d, 2)) + 1
        self.k_min = int(packet_size / (s + 16)) * packet_num
        self.k_max = int(packet_size / (s + 2)) * packet_num
        print("packet size is {}, packet num is {}".format(packet_size, packet_num))
        print("k max is {}, k min is {}".format(self.k_max, self.k_min))
        self.accu_gra = torch.zeros((client_num, total_trainable_params))

    def get_dataLen(self, idx):
        return self.data_len[idx]

    def get_model(self):
        return self.model

    def get_model_gradient(self, idx, args, communication_time, optim=False):
        parameters = torch.tensor([])
        last_parameters = torch.tensor([])
        # for p in self.model.state_dict():
        for p, _ in self.model.named_parameters():
            if p.find("num_batches_tracked") != -1:
                continue
            parameters = torch.cat((parameters, self.model.state_dict()[p].view(-1).detach()))
            last_parameters = torch.cat((last_parameters, self.last_model.state_dict()[p].view(-1).detach()))
        self.accu_gra[idx] = last_parameters - parameters + self.accu_gra[idx]
        
        init_value, init_indices = torch.topk(self.accu_gra[idx].abs(), self.k_max)
        s = int(math.log(self.d, 2)) + 1

        # if communication_time % self.optim_interval == 0:
        if optim:
            self.best_k = 0
            self.best_ks = []
            self.best_bs = []
            self.best_loss = float('inf')
            abs_gra = self.accu_gra[idx].abs()
            sort_gra, _ = torch.sort(abs_gra, descending=True)
            sort_gra = sort_gra * sort_gra
            min_bit = 3
            norm = torch.sum(sort_gra)
            if args['quan'] == 'PQ':
                loss_func = self.PQ_loss
            elif args['quan'] == "QSGD":
                loss_func = self.QSGD_loss
            interval = int((self.k_max - self.k_min) / 40)
            packet_min = int(packet_size / (16 + s))
            packet_max = int(packet_size / (4 + s))

            for k in range(self.k_max, self.k_min, -interval):
                ks = [round(k / packet_num)] * packet_num  # 初始情况下每个包均分参数
                # init_flag = False
                # for _ in range(self.max_init_iter):
                #     packet_mean = int(k / packet_num)
                #     if args['dataset'] == "CIFAR-10":
                #         remain_init_packet = int(packet_num * 0.2)
                #         ks = [random.randint(max(int(packet_mean * 0.5), packet_min),
                #                              min(int(packet_mean * 1.5), packet_max)) for
                #               _ in range(packet_num - remain_init_packet)]
                #     else:
                #         remain_init_packet = int(packet_num * 0.2)
                #         ks = [random.randint(max(int(packet_mean * 0.9), packet_min),
                #                              min(int(packet_mean * 1.2), packet_max)) for
                #               _ in range(packet_num - remain_init_packet)]
                #     if int(packet_size / (32 + s)) <= (k - sum(ks)) / remain_init_packet <= int(packet_size / (2 + s)):
                #         ks.extend([int((k - sum(ks)) / remain_init_packet)] * remain_init_packet)
                #         if args['dataset'] == 'CIFAR-100' and args['quan'] == 'QSGD' and args['iid'] == 'NIID':
                #             ks.sort()
                #         # ks.reverse()
                #         init_flag = True
                #         break
                # if not init_flag:
                #     continue
                bs = [int(packet_size / ks_i - s) for ks_i in ks]
                if bs[0] < min_bit:
                    continue
                print("the parameter number in the packet is {}, the bit in the packet is {}".format(ks[0], bs[0]))

                #  计算固定k后的稀疏化误差
                spar_loss = torch.sum(sort_gra[sum(ks):]) / norm
                print("the k is {}, the spar_loss is {}".format(sum(ks), spar_loss))
                quan_loss = 0
                start = 0
                for i in range(packet_num):
                    quan_loss += (loss_func(ks[i], bs[i], sort_gra[start:start + ks[i]]) / norm)
                    start += ks[i]
                loss = spar_loss + quan_loss
                print("the initial loss is ", loss)

                last_loss = 0
                for _ in range(self.max_optim_iter):
                    kL = 0
                    for i in range(packet_num - 1):
                        kR = kL + ks[i] + ks[i + 1]
                        x, y = ks[i], ks[i + 1]
                        xb, yb = bs[i], bs[i + 1]
                        fx_loss = (loss_func(x, xb, sort_gra[kL: kL + x]) + loss_func(y, yb,
                                                                                      sort_gra[kL + x:kR])) / norm

                        for xb in range(bs[i], 17):
                            x = int(packet_size / (s + xb))
                            y = kR - kL - x
                            if x <= 0 or y <= 0:
                                break
                            yb = int(packet_size / y - s)
                            if yb < min_bit or yb > 17:
                                continue
                            tmp_loss = ((loss_func(x, xb, sort_gra[kL:kL + x])) + (
                                loss_func(y, yb, sort_gra[kL + x:kR]))) / norm
                            if tmp_loss < fx_loss:
                                fx_loss = tmp_loss
                                ks[i], ks[i + 1] = x, y
                                bs[i], bs[i + 1] = xb, yb
                        kL += ks[i]
                    quan_loss = 0
                    accumu_k = 0
                    for i in range(packet_num):
                        quan_loss += (loss_func(ks[i], bs[i], sort_gra[accumu_k:accumu_k + ks[i]]) / norm)
                        accumu_k += ks[i]
                    loss = spar_loss + quan_loss
                    print("the compressed loss after optimizer is ", loss)
                    if loss == last_loss:
                        break
                    last_loss = loss
                if loss < self.best_loss:
                    self.best_k = sum(ks)
                    self.best_ks = ks
                    self.best_bs = bs
                    self.best_loss = loss
            print("the best loss is {} when k is {}".format(self.best_loss, self.best_k))
            print("the best bs is ", self.best_bs)
            print("the best ks is ", self.best_ks)

            final_bs = [self.best_bs[0]]
            self.count = [1]
            for i in range(1, len(self.best_bs)):
                if self.best_bs[i] == final_bs[-1]:
                    self.count[-1] += 1
                else:
                    final_bs.append(self.best_bs[i])
                    self.count.append(1)
            final_ks = []
            for i in range(len(final_bs)):
                final_ks.append(int(packet_size / (s + final_bs[i])) * self.count[i])
            self.best_bs = final_bs
            self.best_ks = final_ks
            self.best_k = sum(self.best_ks)
            print("the best k is {}".format(sum(self.best_ks)))
            print("the best bs is ", self.best_bs)
            print("the best ks is ", self.best_ks)

        # self.unselect_bit = max(int(self.unselect_bit / int(communication_time / self.optim_interval + 1)), 1)
        # print("the bit for unselected parameters is ", self.unselect_bit)

        final_v, i = torch.topk(init_value, self.best_k)
        final_index = init_indices[i]
        unselect_idx = torch.tensor(list(set(np.arange(self.d)) - set(final_index.numpy())))
        # if (communication_time + 1) % 2 == 0:
        #     unselect_idx = unselect_idx[100:]
        print("unselect indices are ", unselect_idx)
        unselect_min = torch.min(self.accu_gra[idx][unselect_idx])
        unselect_max = torch.max(self.accu_gra[idx][unselect_idx])
        unselect_mean = torch.mean(self.accu_gra[idx][unselect_idx].abs())
        unselect_i = 0
        inter = (unselect_max - unselect_min) / (2 ** self.unselect_bit - 1)
        extra_num = 0
        accumu_k = 0
        new_gra = torch.zeros(self.accu_gra[idx].shape)
        new_gra[final_index] = self.accu_gra[idx][final_index]
        
        grad_mean = torch.mean(self.accu_gra[idx].abs())
        grad_max = torch.max(self.accu_gra[idx])
        grad_min = torch.min(self.accu_gra[idx])
        # for i in range(packet_num):
        for i in range(len(self.best_bs)):
            indices = final_index[accumu_k:accumu_k + self.best_ks[i]]
            if args['quan'] == "PQ":
                new_gra[indices] = self.PQ_quan(new_gra[indices], 2 ** (self.best_bs[i]))
            elif args['quan'] == "QSGD":
                new_gra[indices] = self.QSGD_quan(new_gra[indices], 2 ** (self.best_bs[i] - 1))
            self.accu_gra[idx][indices] -= new_gra[indices]
            accumu_k += self.best_ks[i]

            # extras = int((packet_size - self.best_ks[i] * (self.best_bs[i] + s)) / self.unselect_bit)
            # j = unselect_idx[unselect_i: unselect_i + extras]
            # l = ((self.accu_gra[idx][j] - unselect_min) / inter).int() * inter + unselect_min
            # r = l + inter
            # pro = (self.accu_gra[idx][j] - l) / inter
            # seed = torch.rand(self.accu_gra[idx][j].shape)
            # tmp = torch.zeros(self.accu_gra[idx][j].shape)
            # tmp[seed < pro] = r[seed < pro]
            # tmp[seed >= pro] = l[seed >= pro]
            # new_gra[j] = tmp
            # self.accu_gra[idx][j] -= new_gra[j]
            # unselect_i += extras

            # extras = int(packet_size - int(packet_size / (self.best_bs[i] + s)) * (self.best_bs[i] + s)) * self.count[i]
            # print("AAAA ", extras)
            # j = unselect_idx[unselect_i: unselect_i + extras]
            # new_gra[j] = torch.sign(self.accu_gra[idx][j]) * grad_mean * 3
            # self.accu_gra[idx][j] -= new_gra[j]
            # unselect_i += extras
            # extra_num += extras

        
        extras = int(packet_size * remain_packet_num)
        j = unselect_idx[unselect_i: unselect_i + extras]
        new_gra[j] = torch.sign(self.accu_gra[idx][j]) * grad_mean * 3
        self.accu_gra[idx][j] -= new_gra[j]
        unselect_i += extras
        extra_num += extras
        print("the number of extras uploader parameters is ", extra_num)

        return new_gra

    def PQ_quan(self, parameters, centroids):
        # ans = torch.zeros_like(parameters)
        # abs_para = parameters.abs()
        # p_max = torch.max(abs_para)
        # p_min = torch.min(abs_para)
        # interval = (p_max - p_min) / centroids
        # left = ((abs_para - p_min) / interval).int() * interval + p_min
        # right = left + interval
        # probability = (abs_para - left) / (right - left)
        # probability[probability < 1e-5] = 0
        # seed = torch.rand(parameters.shape)
        # ans[seed < probability] = right[seed < probability]
        # ans[seed >= probability] = left[seed >= probability]
        # return ans * torch.sign(parameters)
        
        ans = torch.zeros_like(parameters)
        p_max = torch.max(parameters)
        p_min = torch.min(parameters)
        interval = (p_max - p_min) / (centroids - 1)
        left = ((parameters - p_min) / interval).int() * interval + p_min
        right = left + interval
        probability = (parameters - left) / (right - left)
        probability[probability < 1e-5] = 0
        seed = torch.rand(parameters.shape)
        ans[seed < probability] = right[seed < probability]
        ans[seed >= probability] = left[seed >= probability]
        return ans

    def PQ_loss(self, x, xb, parameter):
        # return x / (2 * ((2 ** xb - 1) ** 2)) * torch.sum(parameter)
        return math.sqrt(x) / (2 ** (2 * xb)) * torch.sum(parameter)

    def QSGD_loss(self, x, xb, parameter):
        # return min(x / (2 ** (2 * xb)), math.sqrt(x) / (2 ** xb)) * torch.sum(parameter)
        return x / (2 ** (2 * xb)) * torch.sum(parameter)

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

    # def get_parameterSize(self, k):
    #     b = 1
    #     s = int(math.log(self.d, 2)) + 1
    #     for b in range(1, 33):
    #         if k * (s + b) > packet_size:
    #             break
    #     return b - 1

    def set_model(self, model):
        self.model.load_state_dict(model.state_dict())
        self.last_model.load_state_dict(model.state_dict())
        # torch.save(model, "net_params.pkl")
        # self.model = torch.load("net_params.pkl")
        # self.last_model = torch.load("net_params.pkl")

    def train(self, idx, communication_time, args):
        self.communication_time[idx] += 1
        print("client no is ", idx, " train num is ", self.communication_time[idx])
        self.model = self.model.to(device)
        self.last_model = self.last_model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        # lr = 0.25 / (math.sqrt(1 + communication_time / 5))
        print("learning rate is ", self.lr)

        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        for epo in range(epoch_num):
            self.model.train()
            for i, (data, label) in enumerate(self.train_loader[idx]):
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                pred = self.model(data)
                loss = loss_fn(pred, label)
                # for p1, p2 in zip(self.model.parameters(), self.last_model.parameters()):
                #     loss += 0.01 / 2 * ((p1-p2)**2).sum()
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print("Train Epoch: {}, iteration: {}, Loss: {}".format(epo, i, loss.item()))

        self.model = self.model.to('cpu')
        self.last_model = self.last_model.to('cpu')
