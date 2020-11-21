#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/16

from sklearn.svm import SVC

from ..base import BaseEstimator
from ..base import ModelMixin


class SVMClassifier(ModelMixin,
                    BaseEstimator):
    # 超参列表
    kernel: str = None

    degree: int = None

    gamma: str = None

    decision_function_shape: str = None

    def fit(self, **kwargs):
        bst = SVC(kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                  decision_function_shape=self.decision_function_shape)

        bst.fit(kwargs.get('x_train'), kwargs.get('y_train'))

        self.model = bst

        return self

    # override
    def predict(self, x):
        return self.model.predict(x)
