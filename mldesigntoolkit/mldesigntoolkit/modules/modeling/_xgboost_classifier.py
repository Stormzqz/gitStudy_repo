#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/10

import xgboost

from ..base import BaseEstimator
from ..base import ModelMixin


class XgboostClassifier(ModelMixin,
                        BaseEstimator):
    # 超参列表
    verbosity: int = 0

    n_jobs: int = -1

    objective: str = 'binary:logistic'

    learning_rate: float = None

    max_depth: int = None

    subsample: float = None

    colsample_bytree: float = None

    colsample_bylevel: float = None

    reg_alpha: float = None

    reg_lambda: float = None

    gamma: float = None

    n_estimators = 1000

    early_stopping_rounds: int = 0

    def fit(self, **kwargs):
        bst = xgboost.XGBClassifier(n_estimators=self.n_estimators, verbosity=self.verbosity, n_jobs=self.n_jobs,
                                    objective=self.objective, learning_rate=self.learning_rate,
                                    max_depth=self.max_depth, subsample=self.subsample,
                                    colsample_bytree=self.colsample_bytree, colsample_bylevel=self.colsample_bylevel,
                                    reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda,
                                    early_stopping_rounds=self.early_stopping_rounds, gamma=self.gamma)

        self.model = bst.fit(kwargs.get('x_train'), kwargs.get('y_train'), eval_set=[(kwargs.get('x_train'), kwargs.get('y_train')), (kwargs.get('x_valid'), kwargs.get('y_valid'))],
                             verbose=False, eval_metric='auc')

        return self
