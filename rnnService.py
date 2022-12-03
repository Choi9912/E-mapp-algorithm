# -*- encoding: utf-8 -*-

import pickle
from datetime import datetime

import torch
import torch.nn as nn
from tensorflow.python.ops.gen_dataset_ops import iterator

from model.base.rnnBase import rnnBase


class rnnService:

    def __init__(self, exec_mode, vocab_size, original, translation, hidden_size, opt, epochs_):
        super(rnnService, self).__init__()
        # 모델
        self.model = None
        # 데이터 셋
        self.dataset = None
        self.exec_mode = exec_mode
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.original = original
        self.translation = translation
        self.opt = opt
        self.epochs = epochs_
    def create_model(self):
        # 인스턴스 생성
        self.model = rnnBase(self.vocab_size, self.hidden_size)
        return self.model

    def get_model(self):

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

    def training(self):

        optimizer = None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        x_ = list(map(ord, "{}".format(self.original)))
        y_ = list(map(ord, "{}".format(self.translation)))
        print("hello -> ", x_)
        print("hola -> ", y_)
        x = torch.LongTensor(x_)
        y = torch.LongTensor(y_)
        rnns = rnnBase(self.vocab_size, self.hidden_size)

        criterion = nn.CrossEntropyLoss().to(device)

        if self.opt in ['SGD', 'Adam', 'AdaGrad']:
            if self.opt == 'SGD':
                optimizer = torch.optim.SGD(rnns.parameters(), lr=1e-3)
            elif self.opt == 'Adam':
                optimizer = torch.optim.Adam(rnns.parameters(), lr=1e-3)
            elif self.opt == 'AdaGrad':
                optimizer = torch.optim.Adagrad(rnns.parameters(), lr=1e-3)
        log = []
        for i in range(self.epochs):
            prediction = rnns(x, y)
            loss = criterion(prediction, y)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            loss_val = loss.data
            log.append(loss_val)
            if i % 100 == 0:
                print("\n 반복: %d 오차: %s" % (i, loss_val.item()))
                _, top1 = prediction.data.topk(1, 1)
                print([chr(c) for c in top1.squeeze().numpy().tolist()])

    def test(self):
        self.model.eval()
        epoch_loss = 0
        x_ = list(map(ord, "{}".format(self.original)))
        y_ = list(map(ord, "{}".format(self.translation)))
        x = torch.LongTensor(x_)
        y = torch.LongTensor(y_)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for i, batch in enumerate(x,y):
                src = batch.src
                trg = batch.trg

                # output: [trg len, batch size, output dim]
                output = self.model(src, trg, 0)  # teacher forcing off
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)  # [(trg len -1) * batch size, output dim]
                trg = trg[1:].view(-1)  # [(trg len -1) * batch size, output dim]

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        print(epoch_loss / len(x))
