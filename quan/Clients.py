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
# import parser
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

epoch_num = 5

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    # transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

packet_size = 1500 * 8
packet_num = 10


class Client():
    def __init__(self, client_num, args, batch_size):
        self.client_num = client_num
        self.data_len = []
        self.communication_time = [0] * self.client_num
        self.train_loader = []
        global packet_num

        if args['dataset'] == "CIFAR-10":
            self.lr = 0.1
            self.batch_size = batch_size
            self.model = m.Net()  # torch.load("../cifar.pkl")  # m.ResNet(m.ResidualBlock)
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
            packet_num = 90
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
            self.batch_size = 256
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

        print("packet size is {}, packet num is {}".format(packet_size, packet_num))

        s = int(math.log(self.d, 2)) + 1
        self.k_min = int(packet_size / (s + 32) * packet_num)
        self.k_max = int(packet_size / (s + 4)) * packet_num
        self.accu_gra = torch.zeros((client_num, total_trainable_params))

    def get_dataLen(self, idx):
        return self.data_len[idx]

    def get_model(self):
        return self.model

    def get_model_gradient(self, idx, args):
        parameters = torch.tensor([])
        last_parameters = torch.tensor([])
        # for p in self.model.state_dict():
        for p, _ in self.model.named_parameters():
            if p.find("num_batches_tracked") != -1:
                continue
            parameters = torch.cat((parameters, self.model.state_dict()[p].view(-1).detach()))
            last_parameters = torch.cat((last_parameters, self.last_model.state_dict()[p].view(-1).detach()))
        self.accu_gra[idx] = (last_parameters - parameters) + self.accu_gra[idx]
        s = int(math.log(self.d, 2)) + 1
        if args['bit'] == 32:
            # k = int(len(last_parameters) * 0.01)
            k = int(packet_size / (s + 32)) * packet_num
            print("the k is ", k)
            value, indices = torch.topk(self.accu_gra[idx].abs(), k)
            new_gra = torch.zeros(self.accu_gra[idx].shape)
            new_gra[indices] = self.accu_gra[idx][indices]
            self.accu_gra[idx][indices] -= new_gra[indices]
        else:
            ks = int(packet_size / (args['bit'] + s))
            # k = ks * (packet_num - int(32 * 2 ** args['bit'] / packet_size + 1))
            k = ks * packet_num
            _, final_index = torch.topk(self.accu_gra[idx].abs(), k)
            print("the number of uploaded parameters is ", k)
            new_gra = torch.zeros(self.accu_gra[idx].shape)
            new_gra[final_index] = self.accu_gra[idx][final_index]

            if args['quan'] == "PQ":
                new_gra[final_index] = self.PQ_quan(new_gra[final_index], 2 ** args['bit'])
            elif args['quan'] == "QSGD":
                new_gra[final_index] = self.QSGD_quan(new_gra[final_index], 2 ** (args['bit'] - 1))
            self.accu_gra[idx][final_index] -= new_gra[final_index]
            # for _ in range(packet_num):
            #     indices = final_index[accumu_k:accumu_k + ks]
            #     if args['quan'] == "PQ":
            #         new_gra[indices] = self.PQ_quan(new_gra[indices], 2 ** args['bit'])
            #     elif args['quan'] == "QSGD":
            #         new_gra[indices] = self.QSGD_quan(new_gra[indices], 2 ** (args['bit'] - 1))
            #     accumu_k += ks
        return new_gra

    def PQ_quan(self, parameters, centroids):
        ans = torch.zeros_like(parameters)
        p_max = torch.max(parameters)
        p_min = torch.min(parameters)
        interval = (p_max - p_min) / (centroids - 1)
        left = ((parameters - p_min) / interval).int() * interval + p_min
        right = left + interval
        probability = (parameters - left) / (right - left)
        seed = torch.rand(parameters.shape)
        ans[seed < probability] = right[seed < probability]
        ans[seed >= probability] = left[seed >= probability]
        return ans

    def QSGD_quan(self, parameters, centroids):
        norm = torch.norm(parameters, p=2)
        rang = parameters.abs() / norm
        l = (rang * centroids).int()
        p = 1 - (rang * centroids - l)
        seed = torch.rand(parameters.shape)
        xi = torch.zeros_like(parameters)
        xi[seed < p] = l[seed < p] / centroids
        xi[seed >= p] = (l + 1)[seed >= p] / centroids
        ans = norm * torch.sign(parameters) * xi
        return ans

    def get_parameterSize(self, k):
        b = 1
        s = int(math.log(self.d, 2)) + 1
        for b in range(1, 8):
            if k * (s + b) + 32 * (2 ** b) > packet_size:
                break
        return b - 1

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

        print("learning rate is ", self.lr)

        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        # if args['dataset'] == 'CIFAR-10':
        #     optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        # else:
        #     optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        # if args['iid'] == 'IID':
        #     optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=0.85)
        # else:
        #     optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=0.85, weight_decay=5e-4)
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
