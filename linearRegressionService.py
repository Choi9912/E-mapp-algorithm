# -*- encoding: utf-8 -*-

import pickle
import sklearn.metrics as metrics
from sklearn.model_selection import KFold, GridSearchCV
from model.base.linearRegressionBase import linearRegressionBase
from sklearn.model_selection import StratifiedKFold
import numpy as np

class linearRegressionService:


    def __init__(self, evaluation_mode, fit_intercept, normalize, copy_X, n_jobs, positive, n_splits, logging):


        super(linearRegressionService, self).__init__()
        self.log = logging

        # common 변수
        self.model = None
        self.dataset = None

        # 모델 특이 변수
        self.evaluation_mode = evaluation_mode
        self.n_splits = n_splits

        # Parameters
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive

    def create_model(self):
        model = linearRegressionBase(self.fit_intercept, self.normalize, self.copy_X, self.n_jobs, self.positive)

        self.model = model.get_model()

    def get_model(self):

        return self.model

    def set_model(self, model):
        self.model = model

    def load_model(self, loadfile):
        ret_model = None
        try:
            loadfp = open(loadfile, 'rb')
            ret_model = pickle.load(loadfp)
        except Exception as ex:
            self.log.print('load_model 오류 : {0}'.format(str(ex)))
        finally:
            return ret_model

    def save_model(self, model, save_file):
        try:
            savefp = open(save_file, 'wb')
            pickle.dump(model, savefp)
        except Exception as ex:
            self.log.print('save_model 오류 : {0}'.format(str(ex)))

    def close_model(self):

        del self.model

    def set_data(self, dataset):

        self.dataset = dataset

    def training(self):

        # Evaluation mode 확인 -- None : train_test_split , not None : Cross_validation(K-fold) + GridSearch
        # 기존 train_test_split 함수는 데이터를 1회 분리하여 모델을 학습하고 검증.
        # Cross Validation --데이터를 여러 번 반복해서 나누고 학습
        if self.evaluation_mode in ["K-fold","StratifiedkFold"]:
            # 모델 파라미터 튜닝을 위한 파라미터 세팅
            param_grid = {'fit_intercept': [self.fit_intercept],
                          'normalize': [self.normalize],
                          'copy_X': [self.copy_X],
                          'n_jobs': [self.n_jobs],
                          'positive': [self.positive]}

            # k-fold cross validation
            if self.evaluation_mode == "K-fold":
                cv = KFold(n_splits=self.n_splits)
            elif self.evaluation_mode == "StratifiedkFold":
                cv = StratifiedKFold(n_splits=self.n_splits)
            # gridsearch
            model_opt = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=cv)
            self.log.print(model_opt)
            # fit
            model_opt.fit(self.dataset['x_train'], self.dataset['y_train'])

            self.log.print('final params', model_opt.best_params_)  # 최적의 파라미터 값 출력
            self.log.print('best score', model_opt.best_score_)  # 최적의 score 값 출력

            my_model = model_opt.best_estimator_

            self.log.print('linearRegressor',
                           'Best Parmeters: {}, Training Accuracy: {:.6f}'.format(model_opt.best_params_,
                                                                                  model_opt.best_score_))
        else:
            my_model = self.model

            # fit
            my_model.fit(self.dataset['x_train'], self.dataset['y_train'])

            # validation set 유무 확인
            if isinstance([], type(self.dataset['x_val'])):
                self.log.print('LinearRegression', 'Training Accuracy: {:.6f}'
                               .format(my_model.score(self.dataset['x_train'], self.dataset['y_train'])))
            else:
                self.log.print('LinearRegression', 'Training Accuracy: {:.6f},  Validation Accuracy: {:.6f}'
                               .format(my_model.score(self.dataset['x_train'], self.dataset['y_train']),
                                       my_model.score(self.dataset['x_val'], self.dataset['y_val'])))

        return my_model

    def test(self):

        y_pred = self.model.predict(self.dataset['x_test'])
        accuracy_score = metrics.accuracy_score(self.dataset['y_test'], y_pred)

        self.log.print('Linear Regression', 'Accuracy: %.2f' % accuracy_score)

        return accuracy_score

    def prediction(self, prediction_val):

        prediction = self.model.predict(prediction_val)

        return prediction
