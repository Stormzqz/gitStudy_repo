#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/1 11:54
# filename: _feature_generate
# software: PyCharm

import numpy as np
from typing import List
from chinese_calendar import is_holiday

from ..base import BaseEstimator
from ..base import HandlerMixin


class TwoDatetimeColumnsSubtractHandler(HandlerMixin,
                                        BaseEstimator):
    """
        功能： 提供两列日期相减的特征生成
        输入： 待处理的数据集 df
        控制： 指定日期特征列表  字符串列表类型 []
               衍生特征列表     字符串列表类型 []
               备注： 可选衍生特征有 year/quarter/month/week/day/age
        输出： 衍生日期格式的数据集 df
    """

    # 指定日期特征列表
    datetime_columns_list: List[str] = []
    # 衍生特征列表
    datetime_features_list: List[str] = []

    def process(self, df):
        if len(self.datetime_columns_list) != 2:
            raise Exception("Please input two column names for datatime subtract!")

        datetime_former = self.datetime_columns_list[0]
        datetime_latter = self.datetime_columns_list[1]
        df[datetime_former] = df[datetime_former].astype('datetime64[ns]')
        df[datetime_latter] = df[datetime_latter].astype('datetime64[ns]')
        join_str = datetime_former + '_' + datetime_latter

        col_list = []
        for feature in self.datetime_features_list:
            feature_column_name = join_str + '_' + feature
            if feature in ['month', 'quarter']:
                df.loc[:, feature_column_name] = (df[datetime_former].dt.year - df[datetime_latter].dt.year) * 12 + \
                                                 (df[datetime_former].dt.month - df[datetime_latter].dt.month)
                df.loc[df[datetime_former].dt.day < df[datetime_latter].dt.day, feature_column_name] -= 1
                if feature == 'quarter':
                    df.loc[:, feature_column_name] = df[feature_column_name].apply(lambda x: int(x / 3))
            elif feature in ['day', 'week']:
                df.loc[:, feature_column_name] = df[datetime_former].__sub__(df[datetime_latter]).dt.days
                if feature == 'week':
                    df.loc[:, feature_column_name] = df[feature_column_name].apply(lambda x: int(x / 7))
            elif feature in ['year', 'age']:
                df.loc[df[datetime_former].dt.month < df[datetime_latter].dt.month, feature_column_name] = \
                    df[datetime_former].dt.year - df[datetime_latter].dt.year - 1
                df.loc[df[datetime_former].dt.month > df[datetime_latter].dt.month, feature_column_name] = \
                    df[datetime_former].dt.year - df[datetime_latter].dt.year

                df.loc[(df[datetime_former].dt.month == df[datetime_latter].dt.month) &
                       (df[datetime_former].dt.day < df[datetime_latter].dt.day), feature_column_name] = \
                    df[datetime_former].dt.year - df[datetime_latter].dt.year - 1
                df.loc[(df[datetime_former].dt.month == df[datetime_latter].dt.month) &
                       (df[datetime_former].dt.day >= df[datetime_latter].dt.day), feature_column_name] = \
                    df[datetime_former].dt.year - df[datetime_latter].dt.year

                df.loc[:, feature_column_name] = df[feature_column_name].apply(lambda x: int(x))
            else:
                raise Exception("The keyword for datetime feature is wrong.")
            col_list.append(feature_column_name)

        return df[col_list]


class DatetimeColumnDecomposeHandler(HandlerMixin,
                                     BaseEstimator):
    """
        功能： 提供日期特征分解功能
        输入： 待处理数据集 df
        控制： 指定日期特征列表  字符串列表类型  []
               衍生特征列表      字符串列表类型  []
               备注：可选衍生特征有 year/quarter/month/week/day/hour/daylight/night/halfyear/
                                    halfmonth/dayofweek/isholiday/isworkday/isweekend
        输出： 衍生日期格式的数据集 df
    """

    # 指定日期特征列表
    datetime_columns_list: List[str] = []
    # 衍生特征列表
    datetime_features_list: List[str] = []

    def process(self, df):
        df = df.copy(deep=True)
        col_list = []
        for datetime_column in self.datetime_columns_list:
            df[datetime_column] = df[datetime_column].astype('datetime64[ns]')
            for datetime_feature in self.datetime_features_list:
                datetime_feature_columnname = datetime_column + '_' + datetime_feature
                if datetime_feature == 'year':
                    df.loc[:, datetime_feature_columnname] = df[datetime_column].dt.year
                elif datetime_feature == 'week':
                    df.loc[:, datetime_feature_columnname] = df[datetime_column].apply(lambda x: int(x.isocalendar()[1]))
                elif datetime_feature in ['month', 'quarter', 'halfyear']:
                    df.loc[:, datetime_feature_columnname] = df[datetime_column].dt.month
                    if datetime_feature == 'quarter':
                        df.loc[:, datetime_feature_columnname] = df[datetime_feature_columnname].apply(
                            lambda x: int((x + 2) / 3))
                    elif datetime_feature == 'halfyear':
                        df.loc[:, datetime_feature_columnname] = df[datetime_feature_columnname].apply(
                            lambda x: int(x < 7))
                elif datetime_feature in ['day', 'halfmonth']:
                    df.loc[:, datetime_feature_columnname] = df[datetime_column].dt.day
                    if datetime_feature == 'halfmonth':
                        df.loc[:, datetime_feature_columnname] = df[datetime_feature_columnname].apply(
                            lambda x: int(x < 16))
                elif datetime_feature in ['hour', 'daylight', 'night']:
                    df.loc[:, datetime_feature_columnname] = df[datetime_column].dt.hour
                    if datetime_feature == 'daylight':
                        df.loc[:, datetime_feature_columnname] = df[datetime_feature_columnname].apply(
                            lambda x: int(6 <= x < 18))
                    elif datetime_feature == 'night':
                        df.loc[:, datetime_feature_columnname] = df[datetime_feature_columnname].apply(
                            lambda x: 1 - int(6 <= x < 18))
                elif datetime_feature in ['dayofweek', 'isworkday', 'isweekend']:
                    df.loc[:, datetime_feature_columnname] = df[datetime_column].dt.dayofweek
                    if datetime_feature == 'isworkday':
                        df.loc[:, datetime_feature_columnname] = df[datetime_feature_columnname].apply(
                            lambda x: int(x < 5))
                    elif datetime_feature == 'isweekend':
                        df.loc[:, datetime_feature_columnname] = df[datetime_feature_columnname].apply(
                            lambda x: 1 - int(x < 5))
                elif datetime_feature == 'isholiday':
                    df.loc[:, datetime_feature_columnname] = df[datetime_column].apply(lambda x: int(is_holiday(x)))
                else:
                    raise Exception("The keyword for datetime feature is wrong.")
                col_list.append(datetime_feature_columnname)

        return df[col_list]


class NumberColumnsCalculatingHandler(HandlerMixin,
                                      BaseEstimator):
    """
        功能： 提供数值特征相加功能
        输入： 待处理的数据集 df
        控制： 指定要相加的特征列表  字符串列表类型  []
              指定计算操作         字符串类型
              备注： 计算操作有 add/subtract/multiply/divide
        输出： 衍生后的数字集 df
    """

    # 指定要计算的数值特征列表
    number_columns_list: List[str] = []
    # 指定计算操作
    calculating_type: str = 'add'

    def process(self, df):
        number_former = self.number_columns_list[0]
        number_latter = self.number_columns_list[1]
        generate_column_name = number_former + '_' + self.calculating_type + '_' + number_latter

        if self.calculating_type == 'add':
            df.loc[:, generate_column_name] = df[number_former] + df[number_latter]
        elif self.calculating_type == 'subtract':
            df.loc[:, generate_column_name] = df[number_former] - df[number_latter]
        elif self.calculating_type == 'multiply':
            df.loc[:, generate_column_name] = df[number_former] * df[number_latter]
        elif self.calculating_type == 'divide':
            df.loc[:, generate_column_name] = df[number_former] / df[number_latter]
            df.loc[df[number_latter] == 0, generate_column_name] = 0
        else:
            raise Exception("This type of calculation is not supported.")

        return df


class CategoryColumnsComposeHandler(HandlerMixin,
                                    BaseEstimator):
    """
        功能： 实现类别特征组合
        输入： 待处理的数据集 df
        控制： 指定需要组合的特征列表   字符串列表类型  []
        输出： 衍生后的数据集
    """

    # 指定需要组合的特征列表
    category_columns_list: List[str] = []

    def process(self, df):
        former = self.category_columns_list[0]
        latter = self.category_columns_list[1]
        generate_column_name = former + '_joinwith_' + latter

        df.loc[:, generate_column_name] = df[former] + '_' + df[latter]

        return df
