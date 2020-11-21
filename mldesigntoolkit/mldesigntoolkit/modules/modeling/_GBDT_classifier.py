#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/10

from sklearn.ensemble import GradientBoostingClassifier

from ..base import BaseEstimator
from ..base import ModelMixin


class GBDTClassifier(ModelMixin,
                     BaseEstimator):
    # 超参列表
    n_estimators: int = 0

    learning_rate: float = 0.1

    max_depth: int = 0

    subsample: float = 0.1

    def fit(self, **kwargs):
        bst = GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                         max_depth=self.max_depth, subsample=self.subsample)

        bst.fit(kwargs.get('x_train'), kwargs.get('y_train'))

        self.model = bst

        return self
