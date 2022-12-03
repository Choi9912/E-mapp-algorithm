# -*- encoding: utf-8 -*-

import numpy as np
import pickle
import sklearn.metrics as metrics
from model.base.decisionTreeClassifierBase import decisionTreeClassifierBase
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold


class decisionTreeClassifierService:


    def __init__(self, evaluation_mode, criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease,
                 class_weight, ccp_alpha, n_splits, logging):


        super(decisionTreeClassifierService, self).__init__()
        self.log = logging

        # common 변수
        self.model = None
        self.dataset = None

        # 모델 특이 변수
        self.evaluation_mode = evaluation_mode
        self.n_splits = n_splits

        # Parameters
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def create_model(self):
        model = decisionTreeClassifierBase(self.criterion,
                                           self.splitter,
                                           self.max_depth,
                                           self.min_samples_split,
                                           self.min_samples_leaf,
                                           self.min_weight_fraction_leaf,
                                           self.max_features,
                                           self.random_state,
                                           self.max_leaf_nodes,
                                           self.min_impurity_decrease,
                                           self.class_weight,
                                           self.ccp_alpha)
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
            param_grid = {'criterion': [self.criterion],
                          'splitter': [self.splitter],
                          'max_depth': [self.max_depth],
                          'min_samples_split': [self.min_samples_split],
                          'min_samples_leaf': [self.min_samples_leaf],
                          'min_weight_fraction_leaf': [self.min_weight_fraction_leaf],
                          'max_features': [self.max_features],
                          'random_state': [self.random_state],
                          'max_leaf_nodes': [self.max_leaf_nodes],
                          'min_impurity_decrease': [self.min_impurity_decrease],
                          'class_weight': [self.class_weight],
                          'ccp_alpha': [self.ccp_alpha]
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

            self.log.print('Decision Tree', 'Best Parmeters: {}, Training Accuracy: {:.6f}'
                           .format(model_opt.best_params_, model_opt.best_score_))

        elif self.evaluation_mode is None:

            my_model = self.model

            # fit
            my_model.fit(self.dataset['x_train'], self.dataset['y_train'])

            # validation set 유무 확인
            if isinstance([], type(self.dataset['x_val'])):
                self.log.print('Decision Tree', 'Training Accuracy: {:.6f}'
                               .format(my_model.score(self.dataset['x_train'], self.dataset['y_train'])))
            else:
                self.log.print('Decision Tree', 'Training Accuracy: {:.6f},  Validation Accuracy: {:.6f}'
                               .format(my_model.score(self.dataset['x_train'], self.dataset['y_train']),
                                       my_model.score(self.dataset['x_val'], self.dataset['y_val'])))
        return my_model

    @staticmethod
    def get_treeInfo(tree, feature_names, model_type):


        # 해당 마디가 leaf node 일 때 negative index를 지닌다.
        left = tree.tree_.children_left
        right = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value
        size = tree.tree_.weighted_n_node_samples
        rule = []

        def recurse(left2, right2, threshold2, features2, node, depth):
            labels = np.asarray([prob_.argmax() for prob_ in value])
            labels = labels[node]
            sample_size = size[node]

            if 'Classifier' in str(tree):
                value_node = None
            else:
                labels = None
                value_node = value[node][0][0]

            if threshold2[node] != -2:
                features_name = features2[node]
                threshold_node = threshold2[node]

                rule.append(
                    (labels, depth, node, features_name, '<=', threshold_node, sample_size, value_node, model_type))

                if left2[node] != -1:
                    recurse(left2, right2, threshold2, features2, left2[node], node)

                if right2[node] != -1:
                    recurse(left2, right2, threshold2, features2, right2[node], node)
            else:
                # leaf_node
                rule.append((labels, depth, node, None, None, None, sample_size, value_node, model_type))

        recurse(left, right, threshold, features, 0, 0)

        return rule

    def test(self):

        y_pred = self.model.predict(self.dataset['x_test'])
        accuracy_score = metrics.accuracy_score(self.dataset['y_test'], y_pred)

        self.log.print('Decision Tree Test', 'Accuracy: %.2f' % accuracy_score)

        return accuracy_score

    def prediction(self, prediction_val):

        prediction = self.model.predict(prediction_val)

        return prediction
