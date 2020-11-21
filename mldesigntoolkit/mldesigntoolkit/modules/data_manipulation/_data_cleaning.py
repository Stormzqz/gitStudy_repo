#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/2 13:53
# filename: _data_cleaning
# software: PyCharm


import warnings
import numpy as np
from pandas import DataFrame
from scipy import stats

from typing import Dict, List
from pydantic import confloat

from ..base import BaseEstimator
from ..base import HandlerMixin
from ..base import RuleMixin


class RowCleaningHandler(HandlerMixin,
                         BaseEstimator):
    """
    行数据清洗
    """
    # 行缺失率阈值
    missing_rate_threshold: confloat(ge=0, le=1) = 0.8

    def process(self, df):
        """
        清洗低质量数据行
        :param df: 待清洗的数据
        :return: 清洗后的数据
        """
        # 数据记录的长度（列数）
        record_length = df.shape[1]

        # 计算每条记录的缺失值数量
        df['_number_missing'] = df.isnull().sum(axis=1)
        # 计算每条记录是否小于阈值的标记
        flag = (df['_number_missing'] / record_length) < self.missing_rate_threshold

        result_df = df[flag]
        # 删除用于计算的字段
        result_df.pop('_number_missing')

        return result_df


class ColumnCleaningRule(RuleMixin,
                         BaseEstimator):
    """
    清洗列缺失率过高的数据
    """
    # 列缺失率阈值
    missing_rate_threshold: confloat(ge=0, le=1) = 0.8
    # 高基数率阈值
    high_cardinality_rate_threshold: confloat(ge=0, le=1) = 0.8
    # 低基数率阈值
    low_cardinality_rate_threshold: confloat(ge=0, le=1) = 0.8
    # 指定列对应的缺失率/高低基数阈值字典
    missing_rate_threshold_dict: Dict[str, confloat(ge=0, le=1)] = {}
    high_cardinality_rate_threshold_dict: Dict[str, confloat(ge=0, le=1)] = {}
    low_cardinality_rate_threshold_dict: Dict[str, confloat(ge=0, le=1)] = {}
    # 保留特征列列表
    keep_columns_list: list = []
    # 数据画像
    data_picture: DataFrame = None

    def fit(self, X_train):
        """
        拟合数据，过滤出不高于各阈值的特征列列表
        :param X_train: 待拟合数据
        :return: 当前类的实例对象
        """
        filtered_columns = []
        data_length = X_train.shape[0]
        column_list = []
        missing_rate_list = []
        high_cardinality_rate_list = []
        low_cardinality_rate_list = []

        for column in X_train.columns:
            column_list.append(column)

            if column in self.keep_columns_list:
                filtered_columns.append(column)
                missing_rate_list.append(0)
                high_cardinality_rate_list.append(0)
                low_cardinality_rate_list.append(0)
            else:
                value_counts = X_train[column].value_counts()
                counts = X_train[column].count()
                if value_counts.empty:
                    # 此情况为空值列，直接过滤掉
                    missing_rate_list.append(1)
                    high_cardinality_rate_list.append(0)
                    low_cardinality_rate_list.append(0)
                    continue
                # 列缺失率计算与过滤
                threshold_of_missing = self.missing_rate_threshold_dict.get(column, self.missing_rate_threshold)
                missing_rate = (data_length - counts) * 1.0 / data_length

                # 列高基数率计算与过滤

                threshold_of_high_cardinality = self.high_cardinality_rate_threshold_dict.get(column,
                                                                                              self.high_cardinality_rate_threshold)
                high_cardinality_rate = value_counts.size * 1.0 / counts

                # 低基数率计算与过滤
                threshold_of_low_cardinality = self.low_cardinality_rate_threshold_dict.get(column,
                                                                                            self.low_cardinality_rate_threshold)
                low_cardinality_rate = value_counts.iloc[0] * 1.0 / counts

                if (missing_rate >= threshold_of_missing) or \
                        (high_cardinality_rate >= threshold_of_high_cardinality) or \
                        (low_cardinality_rate >= threshold_of_low_cardinality):
                    pass
                else:
                    # 保留各值均低于阈值的列
                    filtered_columns.append(column)
                missing_rate_list.append(missing_rate)
                high_cardinality_rate_list.append(high_cardinality_rate)
                low_cardinality_rate_list.append(low_cardinality_rate)
        self.data_picture = DataFrame({'column_name': column_list,
                                       'missing_rate': missing_rate_list,
                                       'high_cardinality_rate': high_cardinality_rate_list,
                                       'low_cardinality_rate': low_cardinality_rate_list})
        self.model = filtered_columns

        return self


class MissingValuesFillingRule(RuleMixin,
                               BaseEstimator):
    """
    数据集缺失值填充
    """
    # 连续型特征缺失值填充方法，默认为空，不处理连续型特征缺失值。
    # 可选参数为：mean（平均数）/mode（众数）/median（中位数）/const_int*（指定常数值）
    filling_continuous_method: str = None
    # 离散型特征缺失值填充方法，默认为空，不处理离散型特征缺失值。
    # 可选参数为：mode（众数）/const_str*（指定常量字符串）
    filling_discrete_method: str = None
    # 指定特征处理方法字典
    # 可选参数为：mean（平均数）/mode（众数）/median（中位数）/const_int/_str*（指定常数值）
    filling_methods_dict: Dict[str, str] = {}

    # 若以上三个控制参数均为空，则不做缺失值填充。

    def fit(self, X_train):
        """
        拟合数据，计算出对应数据填充方法的填充值
        :param X_train: 待拟合数据集
        :return: 当前类的实例对象
        """
        result_dict = {}
        for column in X_train.columns:
            # 获取填充方法
            if str(X_train[column].dtype) in ('object', 'category'):
                filling_method = self.filling_methods_dict.get(column, self.filling_discrete_method)
            else:
                filling_method = self.filling_methods_dict.get(column, self.filling_continuous_method)
            # 计算填充值
            if filling_method == 'mean':
                result_dict[column] = np.mean(X_train[column])
            elif filling_method == 'median':
                result_dict[column] = np.median(X_train[column])
            elif filling_method == 'mode':
                result_dict[column] = X_train[column].value_counts().keys()[0]
            elif filling_method and 'const_' in filling_method:
                result_dict[column] = filling_method[6:]
            else:
                result_dict[column] = None
                warnings.warn("Column '{}' has no value to fill missing_values".format(column))
        self.model = result_dict

        return self

    def transform(self, df):
        """
        执行缺失值填充功能
        :param df: 待处理数据集
        :return: 填充缺失值后的数据集
        """
        for column in df.columns:
            filling_value = self.model.get(column, None)
            # 判断无填充值的情况，给出警告信息
            if not filling_value:
                warnings.warn("Column '{}' has no value to fill missing_values".format(column))
            # 处理category类型数据缺失值填充问题
            if str(df[column].dtype) in ('category', 'str'):
                df[column].astype('str')
                if filling_value:
                    df[column].fillna(str(filling_value), inplace=True)
                df[column].astype('category')
            # 其他类型缺失值直接填充
            else:
                if filling_value:
                    df[column].fillna(filling_value, inplace=True)

        return df


class OutlierHandlingRule(RuleMixin,
                          BaseEstimator):
    """
    数据集异常值替换
    """
    # 待处理连续特征异常值列表
    continuous_columns_list: List[str] = []
    # 待处理离散特征异常值列表
    discrete_columns_list: List[str] = []

    def replace_outliers(self, X_train, col_name, value_max, value_min):
        # 用边界值替换异常值
        index = X_train[col_name] > value_max
        X_train.loc[index, col_name] = value_max
        index = X_train[col_name] < value_min
        X_train.loc[index, col_name] = value_min

        return X_train

    def fit(self, X_train):
        """
        对数值列进行画像
        :param X_train: 待质量画像的数据，要求输入data全部为数值列
        :return: 数据质量画像，最大值、最小值、平均数、方差、中位数、四分位数、正太分布相似度
        """
        handling_list = self.continuous_columns_list + self.discrete_columns_list
        if not handling_list:
            warnings.warn("No feature supplied to fix outliers.")
            empty_df = DataFrame()
            self.model = empty_df
            return self

        numeric_list = X_train.select_dtypes(include=np.number).columns.tolist()
        if set(handling_list) > set(numeric_list):
            raise RuntimeError('Features that to be handled must be numeric type.')

        describe_df = X_train[handling_list].describe().T

        # 循环处理各数值字段
        for column in describe_df.index:
            column_data = X_train[column]

            # 正态分布相似度计算（使用显著水平为15%计算）
            # 参考：https://blog.csdn.net/qq_20207459/article/details/102863982
            # https://www.zhihu.com/question/263864019
            anderson_result = stats.anderson(column_data, 'norm')
            describe_df.loc[column, 'normal'] = anderson_result.statistic < anderson_result.critical_values[0]

            describe_df.loc[column, 'lognormal'] = False
            # 取log后，正态分布相似度计算（使用显著水平为15%计算）
            # 参考同上
            if describe_df.loc[column, 'min'] > 0:
                anderson_log_res = stats.anderson(np.log(column_data), 'norm')
                describe_df.loc[column, 'lognormal'] = anderson_log_res.statistic < anderson_log_res.critical_values[0]

        _ = describe_df.pop('count')
        describe_df['col_name'] = describe_df.index
        self.model = describe_df

        return self

    def transform(self, df):
        """
        执行异常值替换功能
        :param handling_df: 待处理数据集
        :return: 异常值替换后的数据集
        """
        if self.model.shape[0] == 0:
            return df

        self.model = self.model.set_index(self.model['col_name'])

        # 处理连续特征
        for col_name in self.continuous_columns_list:
            # 获取数值边界，用于替换异常值
            std = self.model.loc[col_name, 'std']
            value_max = self.model.loc[col_name, 'mean'] + 3.0 * std
            value_min = self.model.loc[col_name, 'mean'] - 3.0 * std
            # 用边界值替换异常值
            df = self.replace_outliers(df, col_name, value_max, value_min)
        # 处理离散特征
        for col_name in self.discrete_columns_list:
            iqr = self.model.loc[col_name, '75%'] - self.model.loc[col_name, '25%']
            value_max = self.model.loc[col_name, '75%'] + 2.5 * iqr
            value_min = self.model.loc[col_name, '25%'] - 2.5 * iqr
            # 用边界值替换异常值
            df = self.replace_outliers(df, col_name, value_max, value_min)

        return df
