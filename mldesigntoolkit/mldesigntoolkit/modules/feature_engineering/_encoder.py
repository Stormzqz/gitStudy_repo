#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/7 11:44
# filename: _encoder
# software: PyCharm

import pandas as pd
import category_encoders as ce

from typing import Dict, List

from ..util import EncodingMethodEnum
from ..util import SupervisedEncodingMethodEnum
from ..base import BaseEstimator
from ..base import RuleMixin


class Encoder(object):
    """
    编码器父类
    """

    def __init__(self, encoding_method):
        self.encoding_method = encoding_method

    def transform(self, df):
        encoded_df = self.real_encoder.transform(df[self.col_name])
        encoded_df.columns = [col + '_' + self.encoding_method for col in encoded_df.columns]
        return encoded_df


class OnehotEncoder(Encoder):
    """
    onehot 编码器封装
    """

    def fit(self, X, col_name):
        self.col_name = col_name
        self.real_encoder = ce.OneHotEncoder(cols=[col_name]).fit(X[col_name])


class CountEncoder(Encoder):
    """
    频数 编码器封装
    """

    def fit(self, X, col_name):
        self.col_name = col_name
        self.real_encoder = X[col_name].value_counts()

    def transform(self, X):
        encoded_df = pd.DataFrame()
        # 应用value_counts()
        encoded_df[self.col_name + '_count'] = X[self.col_name].apply(lambda x: str(self.real_encoder[x]))
        return encoded_df


class FreqEncoder(Encoder):
    """
    频率 编码器封装
    """

    def fit(self, X, col_name):
        self.col_name = col_name
        self.real_encoder = X[col_name].value_counts()

    def transform(self, X):
        encoded_df = X[[self.col_name]].replace(self.real_encoder).fillna(0)
        count = self.real_encoder.values.sum()
        # 频次计算
        encoded_df[self.col_name] = encoded_df.apply(lambda x: x * 1.0 / count if isinstance(x, int) else 0)
        encoded_df.columns = [self.col_name + '_freq']
        return encoded_df


class CatboostEncoder(Encoder):
    """
    catboost 编码器封装
    """

    def fit(self, X, y, col_name):
        self.col_name = col_name
        self.real_encoder = ce.CatBoostEncoder()
        self.real_encoder.fit(X[col_name], y)


class TargetEncoder(Encoder):
    """
    target 编码器封装
    """

    def fit(self, X, y, col_name):
        self.col_name = col_name
        self.real_encoder = ce.TargetEncoder()
        self.real_encoder.fit(X[col_name], y)


class WoeEncoder(Encoder):
    """
    woe 编码器封装
    """

    def fit(self, X, y, column):
        self.col_name = column
        self.real_encoder = ce.WOEEncoder()
        self.real_encoder.fit(X[column], y)


class EncoderFactory(object):
    def __init__(self, encoding_method):
        self.encoding_method = encoding_method
        encoder_factory = getattr(self, encoding_method, None)
        if encoder_factory:
            self.encoder = encoder_factory()
        else:
            raise TypeError("The given encoding method has no definition.")

    def onehot(self):
        return OnehotEncoder(self.encoding_method)

    def count(self):
        return CountEncoder(self.encoding_method)

    def freq(self):
        return FreqEncoder(self.encoding_method)

    def catboost(self):
        return CatboostEncoder(self.encoding_method)

    def target(self):
        return TargetEncoder(self.encoding_method)

    def woe(self):
        return WoeEncoder(self.encoding_method)


class ColumnEncodingRule(RuleMixin,
                         BaseEstimator):
    """
    功能：提供特无监督征编码功能
    输入：待处理数据集             X_train
    控制：1. 编码方法列表          encoding_methods_list
          2. 指定特征编码方法列表  encoding_methods_dict
          注, 可选编码方法: onehot/count/freq
    输出：编码后的数据集
    """
    # 编码方法列表
    encoding_methods_list: List[EncodingMethodEnum] = [EncodingMethodEnum.onehot]
    # 指定特征编码方法列表
    encoding_methods_dict: Dict[str, List[EncodingMethodEnum]] = {}

    def fit(self, X_train):
        """
        拟合数据，提供有监督/无监督各个编码方法
        :param X_train: 待拟合数据
        :return: 当前类的实例对象
        """
        encoding_model_dict = dict()

        for column in X_train.columns:
            # 初始化字典：key为列名，value为 编码对象列表
            encoding_model_dict[column] = []
            # 获取类别编码方法列表
            method_list = self.encoding_methods_dict.get(column, self.encoding_methods_list)

            # 循环各编码方法
            for method in method_list:
                # 收集 编码方法对象
                encoder = EncoderFactory(method.value).encoder
                encoder.fit(X_train, column)
                encoding_model_dict[column].append(encoder)

        self.model = encoding_model_dict

        return self

    def transform(self, df):
        """
        执行特征编码
        :param df: 代编码数据集
        :return: 编码后的数据集
        """
        # 拆分编码对象列表
        encoding_model_dick = self.model

        # 存储各编码特征dataframe，用于集中拼接
        df_list = []
        # 循环各列特征
        for column in df.columns:
            model_list = encoding_model_dick.get(column, [])
            # 执行特征对应的各个编码方法
            for encoder in model_list:
                encoded_df = encoder.transform(df)
                df_list.append(encoded_df)

        data_encoded = pd.concat(df_list, axis=1)

        return data_encoded


class SupervisedColumnEncodingRule(RuleMixin,
                                   BaseEstimator):
    """
    功能：提供特有监督征编码功能
    输入：待处理数据集：X_train,
          标签数据：y_train
    控制：1. 编码方法列表: encoding_methods_list
          2. 指定特征编码方法列表: encoding_methods_dict
          注, 可选编码方法: catboost/target/woe
    输出：编码后的数据集
    """
    # 编码方法列表
    encoding_methods_list: List[SupervisedEncodingMethodEnum] = [SupervisedEncodingMethodEnum.catboost]
    # 指定特征编码方法列表
    encoding_methods_dict: Dict[str, List[SupervisedEncodingMethodEnum]] = {}

    def fit(self, X_train, y_train):
        """
        拟合数据，提供有监督/无监督各个编码方法
        :param df: 待拟合数据
        :return: 当前类的实例对象
        """
        encoding_model_dict = dict()

        for column in X_train.columns:
            # 初始化字典：key为列名，value为 编码对象列表
            encoding_model_dict[column] = []
            # 获取类别编码方法列表
            method_list = self.encoding_methods_dict.get(column, self.encoding_methods_list)

            # 循环各编码方法
            for method in method_list:
                # 收集 编码方法对象
                encoder = EncoderFactory(method).encoder
                encoder.fit(X_train, y_train, column)
                encoding_model_dict[column].append(encoder)

        self.model = encoding_model_dict

        return self

    def transform(self, df):
        """
        执行特征编码
        :param df: 代编码数据集
        :return: 编码后的数据集
        """
        # 拆分编码对象列表
        encoding_model_dick = self.model

        # 存储各编码特征dataframe，用于集中拼接
        df_list = []
        # 循环各列特征
        for column in df.columns:
            model_list = encoding_model_dick.get(column, [])
            # 执行特征对应的各个编码方法
            for model in model_list:
                encoded_df = model.transform(df)
                df_list.append(encoded_df)

        data_encoded = pd.concat(df_list, axis=1)

        return data_encoded
