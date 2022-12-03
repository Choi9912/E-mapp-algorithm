# -*- encoding: utf-8 -*-

import pickle
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

from model.base.cnnBase import cnnBase


class TrainImageFolder(datasets.ImageFolder):


    def __getitem__(self, index):
        filename = self.imgs[index]
        filename = filename[0].replace('\\', '/')
        filename = filename.split('/')
        classType = filename[filename.__len__() - 2]
        label = self.class_to_idx[classType]
        return super(TrainImageFolder, self).__getitem__(index)[0], label


class TestnPredImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        filename = self.imgs[index]
        filename = filename[0].replace('\\', '/')
        filename = filename.split('/')
        classType = filename[filename.__len__() - 2]
        real_idx = self.class_to_idx[classType]
        return super(TestnPredImageFolder, self).__getitem__(index)[0], real_idx


class cnnService:


    def __init__(self, exec_mode, image_height, image_width, train_dir, test_dir, predic_dir, meanRGB, stdRGB,
                 opt, classes):
        """
        :param image_height: 데이터 input dimension
        :param image_width: 데이터 output dimension
        :param train_dir: 모델 타입명
        :param test_dir: 모델 타입명
        :param predic_dir: 모델 타입명
        :param meanRGB: 모델 타입명
        :param stdRGB: 모델 타입명
        :param opt: 모델 타입명
        :param classes: 모델 타입명
        """

        super(cnnService, self).__init__()
        # 모델
        self.model = None
        # 데이터 셋
        self.dataset = None
        self.exec_mode = exec_mode
        # Image_size
        self.image_height = image_height
        self.image_width = image_width
        self.meanRGB = meanRGB

        self.stdRGB = stdRGB

        self.opt = opt
        # train데이터 위치
        self.train_dir = train_dir
        # test데이터 위치
        self.test_dir = test_dir
        # 예측 대상 데이터 위치
        self.predict_dir = predic_dir
        # 분류 항목 리스트
        self.classes = classes

    def create_model(self):
        # 인스턴스 생성
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = cnnBase().to(device)

        return self.model

    def get_model(self):
        """
        설명 : 모델을 세팅하는 함수
        생성자 : jhchoi
        생성일시 : 2020.09.04
        """

        return self.model

    def set_model(self, model):


        self.model = model

    def load_model(self, loadfile):


        try:
            loadfp = open(loadfile, 'rb')
            self.model = pickle.load(loadfp)
        except Exception as ex:
            create_time = datetime.today().strftime("%Y%m%d%H%M%S")
            print('{0} : load_model 오류 : {1}'.format(create_time, str(ex)))

        return self.model

    def save_model(self, model, save_file):

        try:
            self.model = model
            savefp = open(save_file, 'wb')

            pickle.dump(self.model, savefp)
        except Exception as ex:
            create_time = datetime.today().strftime("%Y%m%d%H%M%S")
            print('{0} : save_model 오류 : {1}'.format(create_time, str(ex)))

    def close_model(self):

        del self.model

    def set_data(self, dataset):

        self.dataset = dataset

    def training(self, batch_size):
        optimizer = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = nn.CrossEntropyLoss().to(device)
        if self.opt in ['SGD', 'Adam', 'AdaGrad']:
            if self.opt == 'SGD':
                optimizer = optim.SGD(self.model.parameters(), lr=0.004)
            elif self.opt == 'Adam':
                optimizer = optim.Adam(self.model.parameters(), lr=0.004)
            elif self.opt == 'AdaGrad':
                optimizer = optim.Adagrad(self.model.parameters(), lr=0.004)

        # transform.Normalize([R 평균, G 평균, B 평균], [R 표준편차, G 표준편차, B 표준편차])
        # 픽셀 정보를 0~255 값을 가지는데 이를 255로 나누면 0.0 ~ 1.0 사이의 값를 가지게 됨.

        normalize = transforms.Normalize(self.meanRGB, self.stdRGB)
        train_loader = data.DataLoader(
            TrainImageFolder(self.train_dir,
                             transforms.Compose([
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
            batch_size=4,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

        for epoch in range(batch_size):
            running_loss = 0.0
            correct = 0
            for i, imge_data in enumerate(train_loader, 0):
                xdata, ydata = imge_data
                xdata, ydata = xdata.to(device), ydata.to(device)

                optimizer.zero_grad()
                outputs = self.model(xdata)
                # print(outputs)
                loss = criterion(outputs, ydata)

                loss.backward()
                optimizer.step()

                running_loss += loss.data
                prediction = torch.max(outputs.data, 1)[1]

                correct += prediction.eq(ydata.data.view_as(prediction)).cpu().sum()

                if (i + 1) % 100 == 0:
                    print('[%d, %5d] loss: %.6f acc : %.6f' % (
                        epoch + 1, i + 1, running_loss / 2000, 100 * correct / ((i + 1) * 4)))
                    running_loss = 0.0

        print('Finished Training')

    def test(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # transform.Normalize([R 평균, G 평균, B 평균], [R 표준편차, G 표준편차, B 표준편차])
        # 픽셀 정보를 0~255 값을 가지는데 이를 255로 나누면 0.0 ~ 1.0 사이의 값를 가지게 됨.
        # transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB])
        normalize = transforms.Normalize(self.meanRGB, self.stdRGB)
        test_loader = data.DataLoader(
            TestnPredImageFolder(self.test_dir,
                                 transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for imge_data in test_loader:
                xdata, ydata = imge_data
                xdata, ydata = xdata.to(device), ydata.to(device)

                outputs = self.model(xdata)
                _, predicted = torch.max(outputs.data, 1)
                total += ydata.size(0)
                correct += (predicted == ydata).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

    def prediction(self, predict_val):

        # 임의의 입력 4를 선언
        new_var = torch.from_numpy(predict_val).float()

        # 입력한 값에 대해서 예측값 y를 리턴받아서 pred_y에 저장
        prediction_y = self.model(new_var)  # forward 연산
        correct_prediction = torch.argmax(prediction_y, 1)
        print("훈련 후 입력이 {0}일 때의 예측값 : {1}".format(predict_val, correct_prediction))

        return correct_prediction


def main():
    if __name__ == "__main__":
        main()
