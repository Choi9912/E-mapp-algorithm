# -*- encoding: utf-8 -*-

from sklearn.linear_model import LogisticRegression


class logisticRegressionBase:


    def __init__(self, penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter,
                 multi_class, verbose, warm_start, n_jobs, l1_ratio):

        super(logisticRegressionBase, self).__init__()

        # 모델을 선언 및 초기화.
        self.model = LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                        intercept_scaling=intercept_scaling, class_weight=class_weight,
                                        random_state=random_state, solver=solver, max_iter=max_iter,
                                        multi_class=multi_class, verbose=verbose, warm_start=warm_start,
                                        n_jobs=n_jobs, l1_ratio=l1_ratio)

    def get_model(self):


        return self.model

    def close(self):

        del self.model
