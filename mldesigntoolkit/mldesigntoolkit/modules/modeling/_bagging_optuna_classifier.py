#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/09

import pandas as pd
from typing import Any

from ..base import BaseEstimator
from ..base import ModelMixin
from ._optuna_classifier import OptunaClassifier
import numpy as np


class BaggingOptunaClassifier(ModelMixin,
                              BaseEstimator):
    variable_dict = {
        'params': {},
        'param_spaces': {},
        'n_splits': 5,
        'n_trials': 20,
        'trial_early_stopping': 4,
        'n_models': 5,
        'random_state': None,
        'classifier': None,
    }

    models = []
    score: Any = None

    # override
    def fit(self, x_train, y_train):

        # 子模型分数列表
        scores = []
        args = self.variable_dict.copy()
        args.pop('n_models')

        # 训练子模型。并记录子模型及其分数
        for i in range(self.variable_dict.get('n_models')):
            # 训练子模型
            cls = OptunaClassifier(**args)
            cls.fit(x_train, y_train)
            self.models.append(cls)
            scores.append(cls.score)
            print("child model no. " + str(i) + " ok, the score is " + str(cls.score))

        # 计算和保存子模型分数平均值
        print("child model scores is :" + str(scores))
        self.score = np.mean(scores)
        return self

    # override
    def predict(self, x):
        y = np.zeros((len(x)))
        for bst in self.models:
            y += bst.predict(x) / self.variable_dict.get('n_models')
        return pd.DataFrame(y, columns=['predict_y'])
