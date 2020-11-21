#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/24 9:40
# filename: _scorecard
# software: PyCharm

import math
import numpy as np
import pandas as pd
from pydantic import confloat

from ..base import BaseEstimator
from ..base import ModelMixin
from . import SKlearnLogisticRegression
from . import StatsLogisticRegression


class ScorecardModel(ModelMixin,
                     BaseEstimator):
    """
    功能：评分卡模型
    输入：训练集自变量表 X_train
          训练集标签 y_train
    控制：指明标签列: label_column
          逻辑回归算法: is_sklearn_LR
          基础分: basic_score
          翻倍分: pdo
          P0: po
          P值筛选阈值：p_value_threshold
    输出：评分卡模型数据
    """
    label_column: str = 'label'
    is_sklearn_LR: bool = False
    basic_score: int = 600
    pdo: int = 20
    p0: confloat(ge=0, le=1) = 0.05
    p_value_threshold: confloat(ge=0, le=1) = 0.05

    p_value_df: pd.DataFrame = None

    def _calculate_scorecard(self, woe_df, model_df):
        # 合并参数矩阵列
        cal_df = woe_df.merge(model_df.loc[:, ['column_name', 'coefficient']], on='column_name')
        # 计算评分
        cal_df['B'] = float(self.pdo) / math.log(2)
        cal_df['A'] = float(self.basic_score) + cal_df['B'] * math.log(float(self.p0))
        cal_df['score'] = round(
            cal_df.loc[:, 'A'] / model_df.shape[0] - cal_df.loc[:, 'B'] * cal_df.loc[:, 'coefficient'] * cal_df.loc[:,
                                                                                                         'encoding'], 0)
        return cal_df

    def fit(self, X_train, y_train):
        """
        特征选择服务
        :param X_train: 数据集
        :return: 评分卡模型
        """
        # 调用逻辑回归模型，获取系数矩阵
        if self.is_sklearn_LR:
            # 拟合模型， 获取参数矩阵
            sklogistic = SKlearnLogisticRegression()
            sklogistic.fit(x_train=X_train, y_train=y_train)
            coefficient_matrix = sklogistic.model.coef_

            # 组织数据
            column_df = pd.DataFrame({'column_name': X_train.columns.tolist()})
            coefficient_df = pd.DataFrame(coefficient_matrix).T
            coefficient_df.columns = ['coefficient']
            model_df = pd.concat([column_df, coefficient_df], axis=1).reset_index(drop=True)
        else:
            # 执行统计包逻辑回归拟合模型， 获取参数矩阵
            # 通过P值筛选数据
            # 嵌套循环用于实现有放回的 P 值校验
            # 外层大循环，针对特征个数循环
            filtered_col_list = X_train.columns.tolist()
            first_level_num = len(filtered_col_list)
            stop_flag = False
            for step_1 in range(first_level_num):
                # 内层循环，实现外层循环特征数量下，有放回的P值校验
                # 加 1 是因为首次循环没有执行特征删除，加 1 后可执行所有特征删除遍历。
                second_level_num = len(filtered_col_list) + 1
                # 各特征 P 值均值series，在内循环中更新
                p_values_series = pd.Series([0.0] * len(filtered_col_list), index=filtered_col_list)
                delete_list = []
                fit_cols_list = filtered_col_list.copy()
                for step_2 in range(second_level_num):
                    # 拟合数据
                    statslogistic = StatsLogisticRegression()
                    statslogistic.fit(x_train=X_train[fit_cols_list], y_train=y_train)
                    # 模型系数及P值
                    coefficient_matrix = statslogistic.model.params
                    p_values = statslogistic.model.pvalues
                    # P值筛选截止条件：所有特征的 P 值均小于给定阈值
                    if step_2 == 0 and p_values.apply(lambda x: x <= self.p_value_threshold).all()  and (coefficient_matrix.apply(lambda x: x >= 0).all() or coefficient_matrix.apply(lambda x: x < 0).all()):
                        stop_flag = True
                        break
                    else:
                        # 更新 P 值series
                        if step_2 == 0:
                            p_values_series = p_values_series.add(p_values)
                        else:
                            _col = (set(p_values_series.index.tolist()) - set(p_values.index.tolist())).pop()
                            fill_v = p_values_series.loc[_col]
                            p_values_series = p_values_series.add(p_values, fill_value=fill_v) / 2
                        # 删除 P 值最大，且没有被删除过的特征
                        sorted_col_list = p_values_series.sort_values(ascending=False).index.tolist()
                        del_col = ''
                        for col in sorted_col_list:
                            if col not in delete_list:
                                del_col = col
                                delete_list.append(col)
                                break
                        # 准备下次循环的特征集，有放回的删除本轮最大 P 值特征
                        if del_col:
                            fit_cols_list = filtered_col_list.copy()
                            fit_cols_list.remove(del_col)
                if stop_flag:
                    break
                else:
                    # 删除 P 均值最大的特征
                    sorted_col = p_values_series.sort_values(ascending=False).index.tolist()
                    if sorted_col:
                        filtered_col_list.remove(sorted_col[0])

            if len(filtered_col_list) == 0:
                raise Exception("No feature's P value is less than the p_value_threshold, please enlarge the threshold."
                                "\n没有特征能够满足 P 值筛选条件，请适当增大 P 值筛选阈值参数: p_value_threshold")

            # 组织数据
            model_df = pd.DataFrame()
            for i in range(len(coefficient_matrix.index)):
                model_df.loc[i, 'column_name'] = coefficient_matrix.index[i]
                model_df.loc[i, 'coefficient'] = coefficient_matrix[i]
            model_df.reset_index(drop=True, inplace=True)

            # 保存各特征显著性数据：p值
            self.p_value_df = p_values.copy(deep=True)
            self.p_value_df = self.p_value_df.to_frame().reset_index()
            self.p_value_df.columns = ['columns', 'p_value']

        # 训练数据表变换
        woe_df = pd.DataFrame()
        for col in X_train[filtered_col_list].columns:
            temp_woe = X_train[col].unique().tolist()
            temp_woe_df = pd.DataFrame({'column_name': [col] * len(temp_woe), 'encoding': temp_woe})
            woe_df = pd.concat([woe_df, temp_woe_df], axis=0).reset_index(drop=True)

        # 计算评分卡
        result_df = self._calculate_scorecard(woe_df, model_df)
        # 特征列各特征值去掉后缀
        result_df.loc[:, 'column_name'] = result_df.loc[:, 'column_name'].apply(
            lambda x: '_'.join(str(x).split('_')[:-1]))

        self.model = result_df

        return self

    def _in_area(self, area, value):
        # 处理分箱为空的情况
        none_list = ['', ' ', 'None', 'nan', 'NaN', 'NULL']
        if str(area) in none_list:
            if str(value) in none_list:
                result = True
            else:
                result = False
        # 处理发分箱特征值匹配
        elif area.startswith('('):
            area = area.replace('(', '').replace(')', '').replace('[', '').replace(']', '').replace(' ', '')
            low_str, high_str = area.split(',')
            low_boundary = -np.inf if low_str == '-inf' else float(low_str)
            high_boundary = np.inf if high_str == 'inf' else float(high_str)
            if low_boundary < float(value) <= high_boundary:
                result = True
            else:
                result = False
        # 处理类别特征匹配（未分箱数据）
        else:
            if area == str(value):
                result = True
            else:
                result = False
        return result

    def _get_score(self, score_dict, value):
        for interval, score in score_dict.items():
            if self._in_area(interval, value):
                return score

    def predict(self, X_test):
        """
        依据模型计算得分
        :param X_test: 数据
        :return: 最终得分
        """
        score_card_df = self.model
        # 过滤特征列
        selected_cols = score_card_df['column_name'].unique().tolist()
        # 处理空值情况
        selected_cols = [item for item in selected_cols if item]
        columns_dict = {}
        for f_col in selected_cols:
            for col in X_test.columns:
                if f_col.startswith(col) or col.startswith(f_col):
                    columns_dict[col] = f_col
                    break
        filter_feature_df = X_test[columns_dict]
        for col in columns_dict.keys():
            # 过滤特征得分分组
            _score = score_card_df.loc[columns_dict[col] == score_card_df['column_name'], ['binning', 'score']]
            # 分箱-得分字典
            map_score_dict = dict(zip(_score['binning'].astype('str').tolist(), _score['score'].tolist()))
            # 将原始数据替换为得分
            filter_feature_df[col] = filter_feature_df[col].apply(lambda x: self._get_score(map_score_dict, x))
        # 计算记录总分
        filter_feature_df['score'] = filter_feature_df[columns_dict.keys()].sum(1)

        filter_feature_df_final = filter_feature_df.loc[:, ['score']]
        return filter_feature_df_final
