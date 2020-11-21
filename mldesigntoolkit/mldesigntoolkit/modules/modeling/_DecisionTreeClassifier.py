#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/16

from sklearn.tree import DecisionTreeClassifier

from ..base import BaseEstimator
from ..base import ModelMixin


class DTClassifier(ModelMixin,
                   BaseEstimator):
    # 超参列表

    max_depth: int = None

    max_leaf_nodes: int = None

    def fit(self, **kwargs):
        bst = DecisionTreeClassifier(max_depth=self.max_depth,
                                     max_leaf_nodes=self.max_leaf_nodes)

        bst.fit(kwargs.get('x_train'), kwargs.get('y_train'))

        self.model = bst

        return self
