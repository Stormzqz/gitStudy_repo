#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/09

import importlib
from pathlib import Path
from typing import Any

from pandas import DataFrame
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import check_random_state

from ..base import BaseEstimator


class Objective(BaseEstimator):
    # 已确定超参
    params: dict = {}
    # 待调优超参
    param_spaces: dict = {}
    # k折
    n_splits: int = 5
    # 模型临时存储目录
    work_dir: Path = None
    # 随机数
    random_state: Any = None
    # 模型名称
    classifier: str = None

    # 训练数据
    x_data: DataFrame = None
    y_data: DataFrame = None

    # 训练集、校验集、测试集
    train_idx_list = []
    valid_idx_list = []
    test_idx_list = []

    def __call__(self, trial):
        """

        :param trial:
        :return:
        """

        # 拷贝已经确定的超参数
        parameters = self.params.copy()

        # 生成需要优化的超参数，目前支持四种类别的超参数控件
        param_spaces = self.param_spaces
        for key, (param_type, param_values) in self.param_spaces.items():
            if param_type == 'loguniform':
                parameters[key] = trial.suggest_loguniform(key, param_values[0], param_values[1])
            elif param_type == 'uniform':
                parameters[key] = trial.suggest_uniform(key, param_values[0], param_values[1])
            elif param_type == 'int':
                parameters[key] = trial.suggest_int(key, param_values[0], param_values[1])
            elif param_type == 'categorical':
                parameters[key] = trial.suggest_categorical(key, param_values)

        if 'bootstrap_type' in parameters.keys():
            if parameters['bootstrap_type'] == 'Bayesian':
                parameters['bagging_temperature'] = trial.suggest_uniform('bagging_temperature', 0, 10)
            elif parameters['bootstrap_type'] == 'Bernoulli':
                parameters['subsample'] = trial.suggest_uniform('subsample', 0.1, 1)

        # 交叉校验结果
        pred_y = self.y_data.copy().astype('float')

        # k折，使用全部训练数据
        random = check_random_state(self.random_state)
        n_splits = self.n_splits
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random)
        for fold, (train_valid_idx, test_idx) in enumerate(cv.split(self.x_data, self.y_data)):
            y_train_valid = self.y_data.iloc[train_valid_idx.tolist()]
            train_idx, valid_idx = train_test_split(train_valid_idx, test_size=1 / (n_splits - 1), shuffle=True,
                                                    random_state=random, stratify=y_train_valid)
            self.train_idx_list.append(train_idx)
            self.valid_idx_list.append(valid_idx)
            self.test_idx_list.append(test_idx)

        for i in range(n_splits):
            # 取数
            train_idx = self.train_idx_list[i]
            valid_idx = self.valid_idx_list[i]
            test_idx = self.test_idx_list[i]
            x_train = self.x_data.iloc[train_idx].reset_index(drop=True)
            x_valid = self.x_data.iloc[valid_idx].reset_index(drop=True)
            y_train = self.y_data.iloc[train_idx].reset_index(drop=True)
            y_valid = self.y_data.iloc[valid_idx].reset_index(drop=True)

            modules = importlib.import_module('mldesigntoolkit.modules.modeling')
            classifier = getattr(modules, self.classifier)

            model_classifier = classifier(**parameters)
            # 训练，使用校验集合确定训练迭代词数
            model_classifier.fit(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)

            #  暂时保存模型
            model_dir = self.work_dir / str(trial.number)
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / ('%d.mll' % i)
            model_classifier.save(str(model_path))
            # 预测测试集合的结果
            p_data = model_classifier.predict(self.x_data.iloc[test_idx])
            pred_y.iloc[test_idx] = DataFrame(index=test_idx, data=p_data)
            print(pred_y.dtypes)
            print(pred_y)

        # 计算auc数值
        score: float = roc_auc_score(self.y_data, pred_y)

        return score
