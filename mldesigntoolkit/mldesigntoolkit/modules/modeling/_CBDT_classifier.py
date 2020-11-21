#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/10
from catboost import CatBoostClassifier

from ..base import BaseEstimator
from ..base import ModelMixin


class CBDTClassifier(ModelMixin,
                     BaseEstimator):
    # 超参列表
    thread_count: int = -1

    objective: str = 'Logloss'

    eval_metric: str = 'AUC'

    verbose: bool = False

    learning_rate: float = None

    depth: int = 0

    colsample_bylevel: float = None

    random_strength: int = None

    l2_leaf_reg: float = None

    bootstrap_type: str = None

    boosting_type: str = None

    n_estimators = 1000

    early_stopping_rounds: int = 5

    def fit(self, **kwargs):
        bst = CatBoostClassifier(n_estimators=self.n_estimators, thread_count=self.thread_count,
                                 objective=self.objective, eval_metric=self.eval_metric, verbose=self.verbose,
                                 learning_rate=self.learning_rate, depth=self.depth,
                                 colsample_bylevel=self.colsample_bylevel, random_strength=self.random_strength,
                                 l2_leaf_reg=self.l2_leaf_reg, bootstrap_type=self.bootstrap_type,
                                 boosting_type=self.boosting_type, early_stopping_rounds=self.early_stopping_rounds)
        self.model = bst.fit(kwargs.get('x_train'), kwargs.get('y_train'), eval_set=[(kwargs.get('x_train'), kwargs.get('y_train')), (kwargs.get('x_valid'), kwargs.get('y_valid'))])

        return self
