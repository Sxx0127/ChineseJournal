import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)


# class Net(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.model1 = nn.Sequential(
#             nn.Conv2d(3, 32, 5, padding=2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, 5, padding=2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5, padding=2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(1024, 64),
#             nn.Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model1(x)
#         return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 96->64 64->128
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
#             nn.Dropout(0.5)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
#             nn.Dropout(0.5)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0),
#             nn.ReLU(),
#             nn.Conv2d(128, 10, kernel_size=(1, 1), padding=0),
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.squeeze(x)
#         return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=2),
#             nn.Dropout(0.5)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=2),
#             nn.Dropout(0.5)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0),
#             nn.ReLU(),
#             nn.Conv2d(128, 10, kernel_size=(1, 1), padding=0),
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.squeeze(x)
#         return x

class FEMNISTNet(nn.Module):
    def __init__(self):
        super(FEMNISTNet, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 32, out_features=256),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=62)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class CNNCifar100(nn.Module):
    def __init__(self):
        super(CNNCifar100, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 4096), nn.ReLU(),  # nn.Dropout(0.5),
            # nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 100)
        )

    def forward(self, x):
        x = self.feature(x)
        output = self.classifier(x)
        return output


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(20 * 7 * 7, 512)  # 两个池化，所以是7*7而不是14*14
        self.fc2 = nn.Linear(512, 10)
        # self.fc3 = nn.Linear(512,cc 10)

    #         self.dp = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 20 * 7 * 7)  # 将数据平整为一维的
        x = F.relu(self.fc1(x))
        #         x = self.fc3(x)
        #         self.dp(x)
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        #         x = F.log_softmax(x,dim=1) NLLLoss()才需要，交叉熵不需要
        return x


class ConvMixer(nn.Module):
    def __init__(self, dim=256, depth=16, kernel_size=5, patch_size=2, n_classes=100):
        super(ConvMixer, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        return self.model(x)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=256),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=64),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


class ALL_CNN(nn.Module):
    def __init__(self):
        super(ALL_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.RReLU(),
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding=1),
            nn.RReLU(),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), padding=1),
            nn.RReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), padding=1),
            nn.RReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), padding=1),
            nn.RReLU(),
            nn.Conv2d(192, 192, kernel_size=(1, 1), padding=0),
            nn.RReLU(),
            nn.Conv2d(192, 10, kernel_size=(1, 1), padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.squeeze(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=100):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.conv5 = conv_block(512, 1028, pool=True)
        self.res3 = nn.Sequential(conv_block(1028, 1028), conv_block(1028, 1028))

        self.classifier = nn.Sequential(nn.MaxPool2d(2),  # 1028 x 1 x 1
                                        nn.Flatten(),  # 1028
                                        nn.Linear(1028, num_classes))  # 1028 -> 100

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out
