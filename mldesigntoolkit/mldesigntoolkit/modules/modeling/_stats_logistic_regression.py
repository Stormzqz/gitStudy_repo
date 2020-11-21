#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/21

import statsmodels.api as sma

from ..base import BaseEstimator
from ..base import ModelMixin


class StatsLogisticRegression(ModelMixin,
                              BaseEstimator):

    def fit(self, **kwargs):
        logistic = sma.Logit(kwargs.get('y_train'), kwargs.get('x_train'))

        self.model = logistic.fit()

        return self

    # override
    def predict(self, x):
        return self.model.predict(x)
