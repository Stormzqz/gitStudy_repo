#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/1 14:13
# filename: _enums
# software: PyCharm

from enum import Enum


class DataTypeEnum(str, Enum):
    """
    指定各数据类型
    """
    int16 = 'int16'
    int32 = 'int32'
    int64 = 'int64'
    float64 = 'float64'
    number = 'number'
    obj = 'object'
    datetime = 'datetime64[ns]'


class MergeMethodEnum(str, Enum):
    """
    指定左右dataframe拼接方法
    """
    left = 'left'
    right = 'right'
    inner = 'inner'
    outer = 'outer'
    index = 'index'


class EncodingMethodEnum(str, Enum):
    """
    指定编码方法
    """
    onehot = 'onehot'
    count = 'count'
    freq = 'freq'


class SupervisedEncodingMethodEnum(str, Enum):
    """
    指定编码方法
    """
    catboost = 'catboost'
    target = 'target'
    woe = 'woe'


class TreeAlgorithmEnum(str, Enum):
    """
    指定编码方法
    """
    XGBoost = 'XGBoost'
    LightGBM = 'LightGBM'
    CatBoost = 'CatBoost'
    GBDT = 'GBDT'
    RF = 'RF'


class BinningMethodEnum(str, Enum):
    """
    指定分箱方法
    """
    EqualWidth = 'uniform'
    EqualFreq = 'quantile'
    KMeans = 'kmeans'


class SupervisedBinningMethodEnum(str, Enum):
    """
    指定分箱方法
    """
    ChiMerge = 'ChiMerge'
    DecisionTree = 'DecisionTree'


class BinningEnum(str, Enum):
    """
    集成有监督与无监督分箱方法
    """
    EqualWidth = BinningMethodEnum.EqualWidth
    EqualFreq = BinningMethodEnum.EqualFreq
    KMeans = BinningMethodEnum.KMeans
    ChiMerge = SupervisedBinningMethodEnum.ChiMerge
    DecisionTree = SupervisedBinningMethodEnum.DecisionTree


if __name__ == "__main__":
    print(DataTypeEnum.int64.value)
    print(MergeMethodEnum.right.value)
