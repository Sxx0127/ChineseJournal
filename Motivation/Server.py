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
import scipy.stats as st
import argparse

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
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

client_num = 100
select_num = 10
batch_size = 50

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
    parse.add_argument('--comp', type=str, default="topk")
    parse.add_argument('--fit', type=int, default=1)
    args = parse.parse_args()
    args.batch_size = batch_size
    
    tmp_model = m.ResNet18()
    model = m.ResNet18() 
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../../CIFAR-10', train=False, download=True,
                       transform=transform_test),
        batch_size=batch_size, shuffle=False
    )
    global_round = 10
    d = 0
    for p in model.state_dict():
    # for p in model.parameters():
        if p.find("num_batches_tracked") != -1:
            continue
        d += model.state_dict()[p].numel()
        # d += p.numel()
    print(f'{d:,} training parameters.')

    clients = Clients.Client(client_num, args)
    communication_time = 0  # 记录当前第几轮通信
    # 一个迭代回合代表一次全局通信
    while True:
        select_clients = random.sample(list(range(client_num)), select_num)

        model_gra = torch.zeros(d)  # 全局梯度

        for i, j in enumerate(select_clients):
            print("the select client num is {}".format(j))
            clients.set_model(model)
            clients.train(j)
            local_model_gra = clients.get_model_gradient(j, args)  # 客户端的时间,同时包括上传\重传请求
            model_gra += 1 / select_num * local_model_gra

        nonzero_indices = model_gra.nonzero()
        if communication_time != 0:
            comment_num = np.intersect1d(nonzero_indices.numpy(), last_indices.numpy())
            print("the number of comment elements is ", len(comment_num))
            print("the proportion of last and current is ", len(comment_num) / last_indices.numel(), 
                  len(comment_num) / nonzero_indices.numel())
            
        last_indices = torch.zeros(nonzero_indices.numel())
        last_indices = nonzero_indices.clone()


        idx = 0
        new_model = model.state_dict()
        for p in new_model:
        # for p in model.parameters():
            if p.find("num_batches_tracked") != -1:
                continue
            size = new_model[p].numel()
            new_model[p].data = new_model[p].data - model_gra[idx:idx + size].reshape(new_model[p].shape).data
            # size = p.numel()
            # p.data = p.data - model_gra[idx:idx + size].reshape(
            #     p.shape).data
            idx += size
        model.load_state_dict(new_model)

        communication_time += 1
        print("Communication Time is ", communication_time)

        if communication_time % 1 == 0:
            acc = 0
            total_loss = 0.
            correct = 0.
            total = 0.
            loss_fn = torch.nn.CrossEntropyLoss()
            model = model.to(device)

            with torch.no_grad():
                model.eval()
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
