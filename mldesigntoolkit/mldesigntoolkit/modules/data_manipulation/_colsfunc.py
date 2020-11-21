#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/1 11:58
# filename: _pandas_io
# software: PyCharm
import datetime
import re
import pandas as pd
import numpy as np

from typing import List, Dict

from ..base import BaseEstimator
from ..base import HandlerMixin
from ..util import DataTypeEnum, MergeMethodEnum


class ColumnConcatingHandler(HandlerMixin,
                             BaseEstimator):
    """
        功能： 级联合并两个数据集
        输入： 待合并的左侧数据集 df_left
               待合并的右侧数据集 df_right
        控制： 指定拼接列名列表 字符串列表类型 join_columns_list   []
               拼接方式         字符串类型     join_method        'left'
        输出： 合并的数据集
    """

    # 指定拼接列名列表 字符串列表类型
    join_column_list: List[str] = []
    # 拼接方式 字符串类型
    join_method: MergeMethodEnum = MergeMethodEnum.index
    # 重复列加后缀
    suffixes: List[str] = ['_left', '_right']

    def process(self, df_left, df_right):
        if self.join_method == MergeMethodEnum.index or len(self.join_column_list) == 0:
            # 按index拼表
            if df_left.shape[0] != df_right.shape[0]:
                raise Exception("records number of the two dataframes are not equal.")
            # df_left.reset_index(drop=True, inplace=True)
            # df_right.reset_index(drop=True, inplace=True)
            df = df_left.merge(df_right, left_index=True, right_index=True, suffixes=self.suffixes)
        else:
            # 依据条件拼表
            df = df_left.merge(df_right, on=self.join_column_list, how=self.join_method, suffixes=self.suffixes)

        return df


class ColumnTypeConvertingHandler(HandlerMixin,
                                  BaseEstimator):
    """
        功能： 列类别转换
        输入： 待转换数据集     df
        控制： 转换列列表       converting_columns_list
               转换类型字符串   to_type
        输出： 列类型转换后的数据集
    """
    # 带转换特征列列表,及要转换成的类型
    converting_dict: dict = {}

    def process(self, df):
        """
        执行数据类型转
        :param df: 输入数据集
        :return: 类型转换后的数据集
        """
        # 如果未指定特征列，则转换所有特征列
        if self.converting_dict:
            # 执行类型转换
            for column in self.converting_dict.keys():
                if self.converting_dict.get(column) in ['date']:
                    df[column] = df[column].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
                elif self.converting_dict.get(column) in ['datetime']:
                    df[column] = df[column].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
                else:
                    df[column] = df[column].astype(self.converting_dict.get(column))
        else:
            pass

        return df


class ColumnTypeFilteringHandler(HandlerMixin,
                                 BaseEstimator):
    """
    功能：根据数据类型筛选数据集
    输入：待过滤数据集       df
    控制：过滤列类型字符串   filtering_type
    输出：按特征类型过滤的数据集
    """
    # 被筛选的数据类型
    filtering_type: DataTypeEnum = DataTypeEnum.obj

    def process(self, df):
        """
        执行数据筛选
        :param df: 输入数据集
        :return: 数据类型筛选后的数据集
        """
        mask = df.astype(str).apply(lambda x: x.str.match(r'(\d{2,4}-\d{2}-\d{2,4})+').all())
        matched_cols = df.columns[mask]
        df[matched_cols] = df[matched_cols].astype('datetime64[ns]')
        return df.select_dtypes(include=self.filtering_type.value)


class ColumnNameFilteringHandler(HandlerMixin,
                                 BaseEstimator):
    """
    功能：根据列名选择数据
    输入：待过滤数据集 df
    控制：过滤列名列表 column_name_list
    输出：按特征名过滤的数据集
    """
    # 需要保留的列list
    column_name_list: List[str] = []

    def process(self, df):
        """
        执行按特征名过滤数据集
        :param df: 输入数据集
        :return: 按特征名过滤后的数据集
        """
        return df[self.column_name_list]


class ColumnRemoveHandler(HandlerMixin,
                          BaseEstimator):
    """
    功能：根据列名删除数据
    输入：待处理数据集 df
    控制：删除列名列表 column_name_list
    输出：删除特征列后的数据集
    """
    # 需要删除的列list
    column_name_list: List[str] = []

    def process(self, df):
        """
        执行删除指定特征
        :param df: 输入数据集
        :return: 删除指定特征后的数据集
        """
        return df.drop(columns=self.column_name_list, axis=1, inplace=False)


class DropAffixHandler(HandlerMixin,
                       BaseEstimator):
    """
    功能：去除数据表中的前缀、后缀
    输入：待处理数据集：df
    控制：去除前缀标记 remove_prefix
          去除后缀标记 remove_suffix
    输出：后缀或前缀去掉后的数据集
    """
    # 去除前缀标记
    remove_prefix: bool = False
    # 去除后缀标记
    remove_suffix: bool = True

    def process(self, df):
        """
        执行词缀去除
        :param df: 输入数据集
        :return: 删除词缀后的数据集
        """
        if self.remove_prefix:
            new_columns = ['_'.join(str(x).split('_')[1:]) if len(str(x).split('_')) > 1 else x for x in df.columns]
            df.columns = new_columns
        if self.remove_suffix:
            new_columns = ['_'.join(str(x).split('_')[:-1]) if len(str(x).split('_')) > 1 else x for x in df.columns]
            df.columns = new_columns

        return df


class DataframesFeatureFilteringHandlar(HandlerMixin,
                                        BaseEstimator):
    """
    功能：使用数据聚集中的特征过滤另一个数据集特征
    输入：参照数据集：compared_df
          带过滤数据集：df
    控制：无
    输出：特征过滤后的数据集
    """

    def process(self, compared_df, df):
        """
        执行数据过滤
        :param compared_df: 输入参照数据集
        :param df: 输入待过滤数据集
        :return: 与参照数据集列数一致的数据集
        """
        compared_df_cols = compared_df.columns.tolist()
        df_cols = df.columns.tolist()
        if set(compared_df_cols) < set(df_cols):
            df_filter = df.loc[:, compared_df_cols]
        else:
            raise RuntimeError("Column names do not match.")

        return df_filter


class ToDataframeHandler(HandlerMixin,
                         BaseEstimator):
    """
    功能：提取模型中的规则数据表
    输入：待处理模型对象：module
    控制：无
    输出：模型组件实例对象中的数据表属性
    """

    def process(self, model_module):
        return model_module.model


class MonotonicitySelectionRuleDataHandler(HandlerMixin,
                                           BaseEstimator):
    """
    功能：整理特征分箱与编码数据，用于点调性特征筛选
    输入：分箱数据 df_binning
          编码数据 df_encoding
    控制：无
    输出：特征，分箱，编码 数据表
    """

    def process(self, binning_df, encoding_df):
        combine_df = []
        encoding_column_list = encoding_df.columns.tolist()
        for binning_col in binning_df.columns:
            # 支持多编码模式
            # original_name = '_'.join(binning_col.split('_')[:-1]) if len(binning_col.split('_')) > 1 else binning_col
            encoding_columns = [col for col in encoding_column_list if binning_col in col]
            for encoding_col in encoding_columns:
                # 合并
                df = pd.concat([binning_df[binning_col], encoding_df[encoding_col]], axis=1)
                # 去重
                df.drop_duplicates(subset=binning_col, keep='first', inplace=True)
                # 处理分箱数据
                if df[binning_col].apply(lambda x: re.match(r'^\(.*\]$', str(x)) is not None).any():
                    # 排序
                    pattern = re.compile(r'[-+]?[0-9]+\.?[0-9]*')
                    df['sorting'] = df[binning_col].apply(
                        lambda x: float(pattern.findall(str(x))[1]) if pattern.findall(str(x)) else np.inf)
                    df.sort_values(by='sorting', inplace=True)
                    df.drop(labels=['sorting'], axis=1, inplace=True)
                    # 修改边界值
                    df[binning_col] = df[binning_col].astype('object')
                    value = pattern.findall(df.iloc[0, 0])[1]
                    min_bin = '(-inf, {}]'.format(value)
                    df.iloc[0, 0] = min_bin
                    # 处理空值问题
                    i = -2 if df.iloc[:, 0].isin([np.nan, 'nan', 'NaN']).any() else -1
                    value = pattern.findall(df.iloc[i, 0])[0]
                    max_bin = '({}, inf)'.format(value)
                    df.iloc[i, 0] = max_bin

                # 表变换
                df['column_name'] = [binning_col] * len(df)
                encoding_method = encoding_col.split('_')[-1]
                df['encoding_method'] = [encoding_method] * len(df)
                df.columns = ['binning', 'encoding', 'column_name', 'encoding_method']
                df = df[['column_name', 'binning', 'encoding', 'encoding_method']]
                combine_df.append(df)
        # 合并
        result_df = pd.concat(combine_df, axis=0)
        return result_df


class ScorecardModelUpdateHandler(HandlerMixin,
                                  BaseEstimator):
    """
    功能：评分卡模型更新组件
    输入：评分卡模型对象 scorecard_model
          待拼接的分箱编码数据集 binning_df
    控制：无
    输出：评分卡模型对象
    """

    def process(self, scorecard_model, binning_df):
        data_left = scorecard_model.model
        data_right = binning_df
        # 按照模型过滤特征数据，同事合并分箱数据
        scorecard_model.model = data_left.merge(data_right, on=['column_name', 'encoding'], how='left')

        return scorecard_model


class GeneratingStdColumnsHandler(HandlerMixin,
                                  BaseEstimator):
    """
    功能：为给定特征组合衍生标准差列
    输入：训练集 X_train
    控制：衍生特征名与参与衍生特征列表字典
    输出：标准差衍生后的数据集
    """
    columns_dict: Dict[str, List[str]]

    def process(self, X_train):
        for name, col_list in self.columns_dict.items():
            X_train[name + '_std'] = X_train[col_list].apply(lambda x: x.std(), axis=1)

        return X_train
