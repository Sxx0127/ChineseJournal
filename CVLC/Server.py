import torch
from torch import nn
from torchvision import datasets, transforms, models
import Clients
import Model as m
import random
import numpy as np
import ssl
import os
import json
import math
import argparse
import scipy.stats as st
from utils import batch_data, word_to_indices, letter_to_vec

ssl._create_default_https_context = ssl._create_unverified_context
myseed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(myseed)
os.environ['PYTHONHASHSEED'] = str(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)

# device = torch.device("cpu")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

client_num = 100
select_num = 10
batch_size = 100

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # R,G,B每层的归一化用到的均值和方差
])
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--iid', type=str, default='IID')
    parse.add_argument('--dataset', type=str, default='CIFAR-10')
    parse.add_argument('--quan', type=str, default="PQ")
    parse.add_argument('--bit', type=int, default=8)
    args = vars(parse.parse_args())
    if args['dataset'] not in ['CIFAR-10', "FEMNIST", "CIFAR-100", "shakespeare"]:
        print("error dataset definition")
        exit(-1)
    if args['iid'] not in ['IID', "NIID"]:
        print("error iid definition")
        exit(-1)
    if args['quan'] not in ["PQ", "QSGD"]:
        print('error quan definition')
        exit(-1)
    if args['bit'] not in [6, 8, 10, 32]:
        print('error bit definition')
        exit(-1)

    if args['dataset'] == "CIFAR-10":
        # batch_size = 75  # lr=0.05,bs=75结果可行
        tmp_model = m.Net()  # torch.load("../cifar.pkl")
        model = m.Net()  # torch.load("../cifar.pkl")  # torch.load("net_params.pkl")
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('~/GraduatedPaper/dataset/CIFAR-10', train=False, download=True,
                             transform=transform_test),
            batch_size=batch_size, shuffle=False
        )
        interval = 5
        global_round = 100
    elif args['dataset'] == "CIFAR-100":
        tmp_model = m.CNNCifar100()
        model = m.CNNCifar100()  # torch.load("net_params.pkl")
        print(model)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('~/GraduatedPaper/dataset/CIFAR-100', train=False, download=True,
                             transform=transform_test),
            batch_size=batch_size, shuffle=False
        )
        interval = 10
        global_round = 180
    elif args['dataset'] == "FEMNIST":
        tmp_model = m.FEMNISTNet()
        model = m.FEMNISTNet()
        data = []
        cnum = 0
        test_data_dir = os.path.join(r'/home/suxiaoxin/GraduatedPaper/dataset/leaf-master/data/femnist/data/test')
        test_files = os.listdir(test_data_dir)
        for f in test_files:
            file_path = os.path.join(test_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            for i in range(len(cdata['users'])):
                client = cdata['users'][i]
                x_test, y_test = cdata['user_data'][client]['x'], cdata['user_data'][client]['y']
                x_test = torch.Tensor(x_test).type(torch.float32)
                y_test = torch.Tensor(y_test).type(torch.int64)
                data.extend([(x.view(1, 28, 28), y) for x, y in zip(x_test, y_test)])
                cnum += 1
                if cnum >= client_num:
                    break
            if cnum >= client_num:
                print("the reading file is ", f)
                break
        len_test = len(data)
        print('the length of test data is ', len_test)
        test_loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size, shuffle=False,
            num_workers=0
        )
        interval = 5
        global_round = 80
    elif args['dataset'] == "shakespeare":
        tmp_model = m.LSTMModel()
        model = m.LSTMModel()  # torch.load("net_params.pkl")
        data = []
        cnum = 0
        test_data_dir = os.path.join('../../leaf-master/data/shakespeare/data/train')
        test_files = os.listdir(test_data_dir)
        for f in test_files:
            file_path = os.path.join(test_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            for i in range(len(cdata['users'])):
                client = cdata['users'][i]
                x_test, y_test = cdata['user_data'][client]['x'], cdata['user_data'][client]['y']
                x = [word_to_indices(word) for word in x_test]
                y = [letter_to_vec(c) for c in y_test]
                x = torch.LongTensor(x)
                y = torch.LongTensor(y)
                data.extend([(w, c) for w, c in zip(x, y)])
                cnum += 1
                if cnum >= client_num:
                    break
            if cnum >= client_num:
                print("the reading file is ", f)
                break
        len_test = len(data)
        print('the length of test data is ', len_test)
        test_loader = torch.utils.data.DataLoader(
            data,
            batch_size=1024, shuffle=False,
            num_workers=0
        )
        interval = 10
        global_round = 180
    # d = sum(model.state_dict()[p].numel() for p in model.state_dict() if p.find("num_batches_tracked") == -1)
    d = sum(p.numel() for p in model.parameters())
    print(f'{d:,} training parameters.')

    # 确定客户端的数量，从而进行数据的划分
    clients = Clients.Client(client_num, args, batch_size)

    communication_time = 0  # 记录当前第几轮通信

    # 一个迭代回合代表一次全局通信
    while True:
        select_clients = random.sample(list(range(client_num)), select_num)
        model_gra = torch.zeros(d)  # 全局梯度
        for i, j in enumerate(select_clients):
            print("the select client num is {}".format(j))
            clients.set_model(model)
            clients.train(j, communication_time, args)
            if i == 0 and communication_time % 50 == 0:
                optim = True
            else:
                optim = False
            local_model_gra = clients.get_model_gradient(j, args, communication_time, optim)  # 客户端的时间,同时包括上传\重传请求
            model_gra += 1 / select_num * local_model_gra

        idx = 0
        # new_model = model.state_dict()
        # for p in new_model:
        #     if p.find('num_batches_tracked') != -1:
        #         continue
        #     size = new_model[p].numel()
        #     new_model[p].data = new_model[p].data - model_gra[idx:idx + size].reshape(new_model[p].shape).data
        #     idx += size
        # model.load_state_dict(new_model)

        for p in model.parameters():
            size = p.numel()
            p.data = p.data - model_gra[idx:idx + size].reshape(p.shape).data
            idx += size


        communication_time += 1
        print("Communication Time is ", communication_time)

        if communication_time % interval == 0:
            acc = 0
            total_loss = 0.
            correct = 0.
            total = 0.
            loss_fn = torch.nn.CrossEntropyLoss()
            model = model.to(device)

            with torch.no_grad():
                # model.eval()
                for (data, target) in test_loader:
                    data = data.to(device)
                    target = target.to(device)

                    output = model(data)
                    total_loss += loss_fn(output, target)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            total_loss /= len(test_loader)
            acc = correct / total * 100.
            print("Communication Time is ", communication_time)
            print("Test loss: {}, Accuracy: {}".format(total_loss, acc))
            model = model.to('cpu')
            if communication_time >= global_round:
                print("Communication Time is ", communication_time)
                break
