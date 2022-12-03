# -*- encoding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier


class decisionTreeClassifierBase:


    def __init__(self, criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
                 min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes, min_impurity_decrease,
                 class_weight, ccp_alpha):


        super(decisionTreeClassifierBase, self).__init__()

        # 모델을 선언 및 초기화.
        self.model = DecisionTreeClassifier(criterion=criterion,
                                            splitter=splitter,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                                            max_features=max_features,
                                            random_state=random_state,
                                            max_leaf_nodes=max_leaf_nodes,
                                            min_impurity_decrease=min_impurity_decrease,
                                            class_weight=class_weight,
                                            ccp_alpha=ccp_alpha)

    def get_model(self):

        return self.model

    def close_model(self):


        del self.model
