# -*- encoding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F


class cnnBase(nn.Module):


    def __init__(self):
        super(cnnBase, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)   # 3 채널을 받아서 64채널로 만들고
        self.pool = nn.MaxPool2d(2, 2)
        self.dout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 64, 3)   # 32채널을 받아서 64채널로 반환하고
        self.conv3 = nn.Conv2d(64, 128, 5)  # 64채널을 받아서 128채널로 반환하고
        self.conv4 = nn.Conv2d(128, 256, 5) # 128채널을 받아서 256채널로 반환하고

        self.fc1 = nn.Linear(21 * 21 * 256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dout(x)
        x = self.pool(F.relu(x))     # pooling을 진행합니다.
        x = self.conv2(x)
        x = self.dout(x)
        x = self.pool(F.relu(x))     # relu에 넣고, pooling을 진행합니다.
        x = self.pool(F.relu(self.conv3(x)))     # relu에 넣고, pooling을 진행합니다.
        x = self.dout(x)
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)   # view는 1차원의 벡터로 변환하는 기능입니다. x.shape[0]는 배치 사이즈와 같습니다.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def print_param(self):


        print('CNN Model Parameters : \n', list(self.model.parameters()))

    def get_model(self):

        return self.__class__

    def close(self):


        del self.__class__
