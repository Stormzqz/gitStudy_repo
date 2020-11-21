#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/09
import contextlib
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from pandas import Series

import optuna
from sklearn.utils import check_random_state

from ..base import BaseEstimator
from ..base import ModelMixin
from ._objective import Objective

import numpy as np


class OptunaClassifier(ModelMixin,
                       BaseEstimator):
    # 已确定超参
    params: dict = {}
    # 待调优超参
    param_spaces: dict = {}
    # k折
    n_splits: int = 5
    # 实验次数
    n_trials: int = 20
    # 实验早停次数
    trial_early_stopping: int = 4
    # 随机数
    random_state: Any = None
    # 模型名称
    classifier: str = None

    # fit方法需要返回的变量
    models = []
    score: float = None
    work_dir: Any = None

    # 创建临时工作目录
    @contextlib.contextmanager
    def dataset(self):
        """
        :return:
        """

        with TemporaryDirectory() as work_dir:
            self.work_dir = Path(work_dir)
            yield

    def fit(self, x_data, y_data):
        """

        :param x_data:
        :param y_data:
        :return:
        """
        random = check_random_state(self.random_state)
        # 优化器的取样器
        sampler = optuna.samplers.TPESampler(seed=random.randint(0, 2 ** 31 - 1))

        # 优化器
        study = optuna.create_study(sampler=sampler, direction='maximize')

        with self.dataset():

            if isinstance(y_data, Series):
                y_data = y_data.to_frame()
            x_data.reset_index(drop=True, inplace=True)
            y_data.reset_index(drop=True, inplace=True)

            args = {
                'params': self.params,
                'param_spaces': self.param_spaces,
                'n_splits': self.n_splits,
                'random_state': self.random_state,
                'classifier': self.classifier,
                'work_dir': self.work_dir,
                'x_data': x_data,
                'y_data': y_data
            }

            objective = Objective(**args)

            # 最大分数
            score_max = -1
            # 当前早停止计数
            early_stop_count = -1

            for i in range(self.n_trials):
                # 实验1次
                study.optimize(objective, n_trials=1)
                # 到目前为止的最大分数
                score = study.best_value
                #  最大分数更新了
                if score > score_max:
                    # 更新最大分数
                    score_max = score
                    # 当前早停止计数设置为0
                    early_stop_count = 0
                # 最大分数伪变化
                else:
                    # 当前早停止计数增1
                    early_stop_count = early_stop_count + 1
                # 判断是否早停
                if early_stop_count > self.trial_early_stopping:
                    break

            # 记录模型的训练分数
            self.score = study.best_value

            # 从工作目录中加载模型
            best_number = study.best_trial.number
            model_dir = self.work_dir / str(best_number)
            models = []

            # 反射获取模型算法对象
            modules = importlib.import_module('mldesigntoolkit.modules.modeling')
            model_classifier = getattr(modules, self.classifier)

            for model_path in model_dir.glob('*.mll'):
                model = model_classifier.load(path=str(model_path))
                models.append(model)
            self.models = models

    def predict(self, x):
        """

        :param x:
        :return:
        """

        # 创建响应变量
        y = np.zeros((len(x)))
        # 累积分折模型((len(X)))
        # 累积分折模型预测结果
        for bst in self.models:
            y += bst.predict(x) / self.n_splits
        return y
