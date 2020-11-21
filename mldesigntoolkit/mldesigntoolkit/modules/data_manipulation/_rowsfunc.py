#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata, Mark
# datetime: 2020/9/2 13:53
# filename: _rowsfunc
# software: PyCharm

import pandas as pd

from typing import Optional
from pydantic import confloat
from sklearn.model_selection import train_test_split

from ..base import BaseEstimator
from ..base import HandlerMixin


class RowPartitioningHandler(HandlerMixin,
                             BaseEstimator):
    """
    功能：按比例分割数据集。
          通常用于分割测试集与训练集，或训练集与验证集
    输入：待分割数据集: df
    控制：1. 指明标签列: label_column
          2. 是否打乱数据: is_shuffle
          3. 是否按标签比例切分: is_stratify
          4. 保持每次分割数据固定: seed
          5. 测试集大小占比: train_rate
    输出：训练集数据：train
          测试集数据：test
    """
    # 指明标签列
    label_column: str = 'label'
    # 是否打乱数据（洗牌），默认是
    is_shuffle: bool = True
    # 依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样。默认是
    is_stratify: bool = True
    # 保证每次分割的两个数据集内容不变。
    seed: Optional[int] = None
    # 训练集大小占比
    train_rate: confloat(ge=0, le=1) = 0.7

    def process(self, df):
        """
        执行数据分割
        :param df: 输入数据
        :return: 分割后的训练集与测试集
        """
        # 分离标签列与数据集
        y = df[self.label_column]

        if self.is_stratify:
            stratify_array = y
        else:
            stratify_array = None

        # 执行数据拆分
        train, test = train_test_split(df, train_size=float(self.train_rate),
                                       random_state=self.seed, shuffle=self.is_shuffle,
                                       stratify=stratify_array)

        return train, test


class RowConcatingHandler(HandlerMixin,
                          BaseEstimator):
    """
    功能：行方向合并两个数据集
    输入：1. 待合并的数据集1: df1
          2. 待合并的数据集2: df2
    控制：无
    输出：合并后的数据集
    """

    def process(self, df1, df2):
        """
        执行两个数据集按行合并
        :param df1: 第一个输入数据集
        :param df2: 第二个输入数据集
        :return: 合并后的数据集
        """
        if df1.shape[0] == 0 and df2.shape[0] == 0:
            df = pd.DataFrame()
        else:
            df = pd.concat([df1, df2], axis=0)
            df = df.reset_index(drop=True)
        return df
