# -*- encoding: utf-8 -*-

from sklearn.linear_model import LinearRegression


class linearRegressionBase:


    def __init__(self, fit_intercept, normalize, copy_X, n_jobs, positive):


        super(linearRegressionBase, self).__init__()
        # 모델을 선언 및 초기화.
        self.model = LinearRegression(fit_intercept=fit_intercept,
                                      normalize=normalize,
                                      copy_X=copy_X,
                                      n_jobs=n_jobs,
                                      positive=positive
                                      )

    def get_model(self):

        return self.model

    def close(self):

        del self.model
