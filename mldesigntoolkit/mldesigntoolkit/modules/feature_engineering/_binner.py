#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/14 11:33
# filename: _binner
# software: PyCharm


import pandas as pd

from typing import Dict, List
from sklearn.preprocessing import KBinsDiscretizer

from ..util import BinningMethodEnum
from ..util import SupervisedBinningMethodEnum
from ..util import ChiMerge
from ..util import DiscretizeByDecisionTree
from ..base import BaseEstimator
from ..base import RuleMixin


class ColumnBinningRule(RuleMixin,
                        BaseEstimator):
    """
    功能：提供无监督特征分箱功能
    输入：待分箱数据集: X_train
    控制：1. 分箱方法列表: binning_method_list
          2. 与分箱方法列表对应的分箱数: binning_number_list
          3. 指定特征分箱方法列表字典: binning_method_dict
          4. 与指定特征分箱方法列表对应的分箱数字典: binning_number_dict
    输出：分箱后的数据集
    """
    # 分箱方法列表
    binning_method_list: List[BinningMethodEnum] = [BinningMethodEnum.EqualFreq]
    # 与分箱方法列表对应的分箱数
    binning_number_list: List[int] = [10]
    # 指定特征分箱方法列表字典
    binning_method_dict: Dict[str, List[BinningMethodEnum]] = {}
    # 与指定特征分箱方法列表对应的分箱数字典
    binning_number_dict: Dict[str, List[int]] = {}

    def fit(self, X_train):
        """
        数值列自动分箱服务
        :param X_train: 待分箱的数据
        :return: 分箱模型字典
        """
        dict_of_binning_model = dict()
        dict_of_binning_name = dict()

        # 循环各列
        for col_name in X_train.columns:
            # 声明字典value为列表
            dict_of_binning_model[col_name] = []
            dict_of_binning_name[col_name] = []
            # 获取分箱方法
            method_list = self.binning_method_dict.get(col_name, self.binning_method_list)
            number_list = self.binning_number_dict.get(col_name, self.binning_number_list)

            # 获取 箱数 与 分箱策略
            for i in range(len(method_list)):
                method = method_list[i]
                number = number_list[i]
                # 创建分箱对象：KBinsDiscretizer
                model = KBinsDiscretizer(n_bins=number, encode='ordinal', strategy=method).fit(X_train[[col_name]])
                # 将分箱对象与分箱方法保存到对应字典
                dict_of_binning_name[col_name].append(method)
                dict_of_binning_model[col_name].append(model)

        self.model = [dict_of_binning_name, dict_of_binning_model]
        return self

    def transform(self, df):
        """
        数值列自动应用分箱服务
        :param df: 待分箱的数据
        :return: 分箱后的数据
        """
        # 获取分箱方法名
        dict_of_binning_name = self.model[0]

        # 获取分箱对象：KBinsDiscretizer
        dict_of_binning_model = self.model[1]
        df_list = []
        # 循环各列
        for col_name in df.columns:
            # 获取分箱对象和名称列表
            model_list = dict_of_binning_model.get(col_name, [])
            name_list = dict_of_binning_name.get(col_name, [])

            # 循环模型列表
            for i in range(len(model_list)):
                model = model_list[i]
                name = name_list[i]
                # 执行分箱
                binned_data = model.transform(df[[col_name]])
                binned_df = pd.DataFrame(binned_data, columns=[col_name + "_" + name])
                # 添加后缀：S，分箱后转类别
                # binned_df[col_name + "_" + name] = binned_df[col_name + "_" + name].map(lambda x: str(x) + 'S')
                # 收集分箱数据
                df_list.append(binned_df)
        result_df = pd.concat(df_list, axis=1)
        return result_df


class SupervisedColumnBinningRule(RuleMixin,
                                  BaseEstimator):
    """
    功能：提供有监督特征分箱功能
    输入：待分箱数据集 X_train
          标签数据     y_train
    控制：1. 指明标签列：label_column
          2. 分箱方法列表: binning_method_list
          3. 与分箱方法列表对应的分箱数: binning_number_list
          4. 指定特征分箱方法列表字典: binning_method_dict
          5. 与指定特征分箱方法列表对应的分箱数字典: binning_number_dict
    输出：分箱后的数据集
    """
    # 指明标签列
    label_column: str = 'label'
    # 分箱方法列表
    binning_method_list: List[SupervisedBinningMethodEnum] = [SupervisedBinningMethodEnum.ChiMerge]
    # 分箱内最少数据量阈值
    bin_size_threshold: int = 5
    # 指定特征分箱方法列表字典
    binning_method_dict: Dict[str, List[SupervisedBinningMethodEnum]] = {}
    # 各分箱算法的分箱数列表（决策树为深度）
    binning_number_list: List[int] = [10]
    max_depth_list: List[int] = [10]
    min_samples_leaf: float = 0.05
    max_leaf_nodes: int = 6
    tree_model: str = 'gini'

    def fit(self, X_train, y_train):
        """
        数值列自动分箱服务
        :param X_train: 待分箱的数据
        :param y_train: 标签列名称
        :return: 分箱模型字典
        """
        binning_model_dict = dict()
        binning_name_dict = dict()

        # 循环各列
        for colName in X_train.columns:
            # 声明字典value为列表
            binning_model_dict[colName] = []
            binning_name_dict[colName] = []

            # 获取分箱方法
            model_list = self.binning_method_dict.get(colName, self.binning_method_list)

            # 获取 箱数 与 分箱策略
            # 卡方分箱的data中必须包含标签列；决策树的data中不能包含标签列
            for binning in model_list:
                if binning.value == 'ChiMerge':
                    for number in self.binning_number_list:
                        df = pd.concat([X_train, y_train], axis=1)
                        model = ChiMerge(column=colName, bin_size_threshold=self.bin_size_threshold,
                                         num_of_bins=number).fit(df=df[[colName, self.label_column]],
                                                                 label=self.label_column)
                        binning_model_dict[colName].append(model)
                        binning_name_dict[colName].append(binning.value + str(number))

                elif binning.value == 'DecisionTree':
                    for number in self.max_depth_list:
                        model = DiscretizeByDecisionTree(col=colName, min_samples_leaf=self.min_samples_leaf,
                                                         max_leaf_nodes=self.max_leaf_nodes, tree_model=self.tree_model,
                                                         max_depth=number).fit(X_train, y_train)
                        binning_model_dict[colName].append(model)
                        binning_name_dict[colName].append(binning.value + str(number))
                else:
                    raise Exception("error method")

        self.model = [binning_model_dict, binning_name_dict]

        return self

    def transform(self, X_train):
        """
        数值列自动应用分箱服务
        :param X_train: 待分箱的数据
        :return: 分箱后的数据
        """
        # 获取分箱对象
        binning_model_dict = self.model[0]
        # 获取分箱方法名
        binning_name_dict = self.model[1]

        df_list = []
        # 循环各列
        for column in X_train.columns:
            # 获取分箱对象和名称列表
            model_list = binning_model_dict.get(column, [])
            name_list = binning_name_dict.get(column, [])

            length = len(model_list)
            # 循环模型列表
            for i in range(length):
                model = model_list[i]
                name = name_list[i]
                # 执行分箱
                data_binned = model.transform(X_train[[column]])
                data_binned_df = data_binned.drop(columns=column, inplace=False).astype('str')
                data_binned_df.columns = [column + "_" + name]
                # 收集分箱数据
                df_list.append(data_binned_df)

        result_df = pd.concat(df_list, axis=1)
        return result_df
