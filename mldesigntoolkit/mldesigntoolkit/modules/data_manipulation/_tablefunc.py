#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/4 16:03
# filename: _tablefunc
# software: PyCharm

from typing import Dict, List

from ..base import BaseEstimator
from ..base import RuleMixin


class HighAndLowCardinalitySplittingRule(RuleMixin,
                                         BaseEstimator):
    """
    功能：按高低基数阈值分割数据集
    输入：待处理数据集                              X_train
    控制：1. 高低基数分割阈值                       cardinality_threshold
          2. transform返回高基数据集或者低基数据集  is_high_cardinality
    输出：
    """
    # 高低基数分割阈值
    cardinality_threshold: int = 5
    # transform方法返回高基数据集或者低基数据集
    is_high_cardinality: bool = True

    def fit(self, X_train):
        """
        按基数拟合数据集各特征
        :param X_train: 待拟合数据集
        :return: 当前类的实例对象
        """
        high_cardinality_list = []
        low_cardinality_list = []
        # 按阈值分割高低基特征
        for column in X_train.columns:
            value_counts = X_train[column].value_counts()
            if len(value_counts) > self.cardinality_threshold:
                high_cardinality_list.append(column)
            else:
                low_cardinality_list.append(column)
        # 组织模型数据
        result_dict = {'high_cardinality_cols': high_cardinality_list, 'low_cardinality_cols': low_cardinality_list}
        self.model = result_dict

        return self

    def transform(self, df):
        """
        执行数据集高低基数据分割
        :param df: 待处理数据集
        :return: 分割后的高基或低基数据集
        """
        if self.is_high_cardinality:
            result_columns = self.model['high_cardinality_cols']
        else:
            result_columns = self.model['low_cardinality_cols']

        return df[result_columns]
