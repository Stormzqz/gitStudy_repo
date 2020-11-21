#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/16

from sklearn.ensemble import RandomForestClassifier

from ..base import BaseEstimator
from ..base import ModelMixin


class RFClassifier(ModelMixin,
                   BaseEstimator):
    # 超参列表
    n_estimators: int = None

    n_jobs: int = None

    max_depth: int = None

    max_leaf_nodes: int = None

    def fit(self, **kwargs):
        bst = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, n_jobs=self.n_jobs,
                                     max_leaf_nodes=self.max_leaf_nodes)

        bst.fit(kwargs.get('x_train'), kwargs.get('y_train'))

        self.model = bst

        return self
