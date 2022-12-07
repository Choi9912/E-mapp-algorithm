# -*- encoding: utf-8 -*-

import pickle
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from model.base.logisticRegressionBase import logisticRegressionBase


class logisticRegressionService:


    def __init__(self, evaluation_mode, penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight,
                 random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio,
                 n_splits, logging):

        super(logisticRegressionService, self).__init__()
        self.log = logging

        # common 변수
        self.model = None
        self.dataset = None

        # 모델 특이 변수
        self.evaluation_mode = evaluation_mode
        self.n_splits = n_splits

        # Parameters
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def create_model(self):
        model = logisticRegressionBase(self.penalty, self.dual, self.tol, self.C,
                                   self.fit_intercept, self.intercept_scaling, self.class_weight,
                                   self.random_state, self.solver, self.max_iter,
                                   self.multi_class, self.verbose, self.warm_start, self.n_jobs, self.l1_ratio)

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
        if self.evaluation_mode in ["K-fold","StratifiedkFold"]:            # 모델 파라미터 튜닝을 위한 파라미터 세팅
            param_grid = {'penalty': [self.penalty], 'dual': [self.dual], 'tol': [self.tol], 'C': [self.C],
                          'fit_intercept': [self.fit_intercept],
                          'intercept_scaling': [self.intercept_scaling], 'class_weight': [self.class_weight],
                          'random_state': [self.random_state],
                          'solver': [self.solver], 'max_iter': [self.max_iter], 'multi_class': [self.multi_class],
                          'verbose': [self.verbose], 'warm_start': [self.warm_start],
                          'n_jobs': [self.n_jobs], 'l1_ratio': [self.l1_ratio]
                          }

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

            self.log.print('logisticRegressor',
                           'Best Parmeters: {}, Training Accuracy: {:.6f}'.format(model_opt.best_params_,
                                                                                  model_opt.best_score_))
        else:
            my_model = self.model

            # fit
            my_model.fit(self.dataset['x_train'], self.dataset['y_train'])

            # validation set 유무 확인
            self.log.print('Logistic Regression', 'Training Accuracy: {:.6f},  Validation Accuracy: {:.6f}'
                           .format(my_model.score(self.dataset['x_train'], self.dataset['y_train']),
                                   my_model.score(self.dataset['x_val'], self.dataset['y_val'])))

        return my_model

    def test(self):

        prediction = self.model.score(self.dataset['x_test'], self.dataset['y_test'])

        self.log.print('Logistic Regression Test', 'Accuracy: %.2f' % prediction)

    def prediction(self, prediction_val):

        prediction = self.model.predict(prediction_val)

        return prediction
