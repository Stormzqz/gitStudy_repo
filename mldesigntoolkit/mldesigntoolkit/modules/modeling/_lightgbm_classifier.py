#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/10

import lightgbm

from ..base import ModelMixin
from ..base import BaseEstimator


class LightGBMClassifier(ModelMixin,
                         BaseEstimator):
    # 超参列表
    verbosity: int = 0

    n_jobs: int = -1

    objective: str = 'binary:logistic'

    metric: str = 'auc'

    boosting_type: str = 'gbrt'

    learning_rate: float = None

    num_leaves: int = None

    min_child_sample: int = None

    bagging_freq: int = None

    feature_fraction: float = None

    bagging_fraction: float = None

    early_stopping_rounds: int = 5

    def fit(self, **kwargs):
        d_train = lightgbm.Dataset(kwargs.get('x_train'), kwargs.get('y_train'))

        d_valid = lightgbm.Dataset(kwargs.get('x_valid'), kwargs.get('y_valid'))

        parameters = {
            'vervosity': self.verbosity,
            'n_jobs': self.n_jobs,
            'objective': self.objective,
            'mestric': self.metric,
            'boosting_type': self.boosting_type,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'min_child_sample': self.min_child_sample,
            'bagging_freq': self.bagging_freq,
            'feature_fraction': self.feature_fraction,
            'bagging_fraction': self.bagging_fraction
        }

        self.model = lightgbm.train(parameters, d_train, valid_sets=[d_train, d_valid], valid_names=['train', 'valid'],
                                    num_boost_round=1000, verbose_eval=1000,
                                    early_stopping_rounds=self.early_stopping_rounds)

        return self

    # override
    def predict(self, x):
        return self.model.predict(x)
