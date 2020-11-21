#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/21

from sklearn.linear_model import LogisticRegressionCV

from ..base import BaseEstimator
from ..base import ModelMixin


class SKlearnLogisticRegression(ModelMixin,
                                BaseEstimator):

    def fit(self, **kwargs):
        sklogistic = LogisticRegressionCV(class_weight='balanced')

        self.model = sklogistic.fit(kwargs.get('x_train'), kwargs.get('y_train'))

        return self

    # override
    def predict(self, x):
        return self.model.predict(x)
