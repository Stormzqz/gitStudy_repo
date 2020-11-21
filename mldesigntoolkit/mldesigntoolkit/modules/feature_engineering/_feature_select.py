#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/8 17:18
# filename: _feature_select
# software: PyCharm

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pandas import DataFrame, Series
from typing import List
from pydantic import confloat
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from ..util import TreeAlgorithmEnum
from ..modeling import ScorecardModel
from ..base import BaseEstimator
from ..base import RuleMixin


def _calc_woe_iv(df, col, target):
    """
    计算指标的WOE和IV值
    :param df: dataframe containing feature and target
    :param col: the feature that needs to be calculated the WOE and iv, usually categorical type
    :param target: good/bad indicator
    :return: WOE and IV in a dictionary
    """
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    bad.fillna(0, inplace=True)
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: (x * 1.0 + 0.0000001) / (B + 0.0000001))
    regroup['good_pcnt'] = regroup['good'].map(lambda x: (x * 1.0 + 0.0000001) / (G + 0.0000001))
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    WOE_dict = regroup[[col, 'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV': IV}


def _judge_monotonicity(encoding_group, encoding):
    # 剔出“空值分箱”，空分箱不参与单调性判断
    judge_df = encoding_group.copy(deep=True)
    if 'binning' in judge_df.columns.to_list():
        if judge_df['binning'].tolist()[-1] in [np.nan, 'nan', 'NaN']:
            judge_df = judge_df.iloc[:-1, :]
    # 分箱数大于等于 4 时执行单调性判断
    if len(judge_df.loc[:, encoding]) >= 4:
        # 用类似于 2 阶导数的方法判断单调性
        encoding_list = judge_df.loc[:, encoding].values.tolist()

        # 计数类似于一阶导数的数值列表，递增与平行值为1，递减为-1
        derivative_list = []
        for i in range(len(encoding_list) - 1):
            if encoding_list[i + 1] >= encoding_list[i]:
                derivative_list.append(1)
            else:
                derivative_list.append(-1)
        # 计算类似于二阶导数的数值列表，“一阶导数”不变时为0，变化时（出现单调性变化）为1
        second_derivative_list = []
        for j in range(len(derivative_list) - 1):
            if derivative_list[j + 1] - derivative_list[j] == 0:
                second_derivative_list.append(0)
            else:
                second_derivative_list.append(1)

        # 判断二阶导数，点调性变化最多只能出现一次，否则剔出此特征
        # 单调性过滤仅保留 递增，递减 或者 出现过一次单调性变化的特征
        if second_derivative_list.count(0) >= len(encoding_list) - 3:
            judgement = 1
        else:
            judgement = 0
        encoding_group['judgement'] = judgement
    # 分箱数为 2 或 3 时，默认保留此特征
    elif (len(judge_df.loc[:, encoding]) == 2) or (len(judge_df.loc[:, encoding]) == 3):
        encoding_group['judgement'] = 1
    # 分箱数为 1 的情况，剔出此特征
    else:
        encoding_group['judgement'] = 0
    return encoding_group


class TreeModelFactory(object):
    def __init__(self, algorithm):
        model_factory = getattr(self, algorithm, None)
        if model_factory:
            self.model = model_factory()
        else:
            raise TypeError("The given tree algorithm name has no definition.")

    def GBDT(self):
        return GradientBoostingClassifier()

    def RF(self):
        return RandomForestClassifier()

    def XGBoost(self):
        return xgb.XGBClassifier()

    def LightGBM(self):
        return lgb.LGBMClassifier()

    def CatBoost(self):
        return CatBoostClassifier()


class FeatureSelectionByTreeAlgorithmRule(RuleMixin,
                                          BaseEstimator):
    """
    功能： 依据树模型特征重要性，提供特征筛选功能
    输入： 待处理数据集 X_train
           标签数据 y_train
    控制： 重要性阈值: importance_threshold
           topn参数：top_n
           指定筛选算法 filtering_algorithm
           备注：可选特征筛选方法: XGBoost/LightGBM/CatBoost/GBDT/RF
    输出： 特征筛选后的数据集
    """
    # 指定筛选算法
    filtering_algorithm: TreeAlgorithmEnum = TreeAlgorithmEnum.XGBoost
    # 保留特征列表
    keep_columns_list: List[str] = []
    # 重要性阈值
    importance_threshold: int = 0
    # 保留最重要的特征个数
    top_n: int = 100

    def fit(self, X_train, y_train):
        """
`       拟合数据，通过model.feature_importances_过滤特征。
        :param df: 待拟合数据集
        :return: 当前类的实例对象
        """
        # 各个模型训练，调用fit方法
        model = TreeModelFactory(self.filtering_algorithm.value).model
        model.fit(X_train, y_train)

        feature_and_importance_list = list(zip(X_train.columns.tolist(), model.feature_importances_.tolist()))
        feature_and_importance_list.sort(key=lambda x: x[1], reverse=True)

        result_features = []
        for i in range(0, min(len(feature_and_importance_list), self.top_n)):
            if feature_and_importance_list[i][0] in self.keep_columns_list:
                result_features.append(feature_and_importance_list[i][0])
            elif feature_and_importance_list[i][1] > self.importance_threshold:
                result_features.append(feature_and_importance_list[i][0])

        self.model = result_features

        return self


class FeatureSelectionByCorrelationRule(RuleMixin,
                                        BaseEstimator):
    """
    功能： 依据自变量之间和自变量与因变量相关性，提供特征筛选功能
    输入： 待处理数据集 X_train
           标签数据 y_train
    控制： 1. 自变量间筛选阈值: xx_correlation_threshold
           2. 自变量因变量间筛选阈值: xy_correlation_threshold
           3. 筛选最大特征数阈值: max_column_number_threshold
           4. 保留特征列表: keep_columns_list
    输出： 特征筛选后的数据集
    """
    # 指定标签列名
    label_name: str = 'label'
    # 自变量间相关性阈值(大于，保留与label相关性高的一个)
    xx_correlation_threshold: confloat(ge=0, le=1) = 0.7
    # 自变量与因变量相关性阈值(小于，排除)
    xy_correlation_threshold: confloat(ge=0, le=1) = 0.1
    # 最大特征数阈值
    max_column_number_threshold: int = 200
    # 保留特征列表
    keep_columns_list: List[str] = []
    # 保留变量间相关性
    xx_corr: pd.DataFrame = None

    def fit(self, X_train, y_train):
        """
        拟合数据，计算相关性，过滤特征
        :param df: 待拟合数据集
        :return: 当前类的实例对象
        """

        # 变量间的相关性("pearson-皮尔逊"相关系数:(ρX,Y)等于它们之间的协方差cov(X,Y)除以它们各自标准差的乘积(σX, σY))
        xx_correlation = X_train.corr()
        xx_correlation_columns = xx_correlation.columns.tolist()
        # 转成上三角矩阵
        xx_correlation_up_matrix = np.triu(xx_correlation, 1)
        # 组织为dataframe
        xx_correlation_up_df = pd.DataFrame(xx_correlation_up_matrix, columns=xx_correlation_columns,
                                            index=xx_correlation_columns)
        # 将数据的列“旋转”为行，转成n*n行，3列的矩阵（unstack：将数据的行“旋转”为列）
        stack_xx_correlation_df = xx_correlation_up_df.stack()
        feature_base = [stack_xx_correlation_df.index[i][0] for i in range(len(stack_xx_correlation_df.index))]
        feature_compare = [stack_xx_correlation_df.index[i][1] for i in range(len(stack_xx_correlation_df.index))]
        xx_correlation_df = pd.DataFrame({'feature_base': feature_base, 'feature_compare': feature_compare,
                                          'correlation': stack_xx_correlation_df.values})
        # 删除下三角，既相关性系数标记为0的记录
        xx_correlation_df = xx_correlation_df[xx_correlation_df.correlation != 0]

        self.xx_corr = xx_correlation_df

        # 自变量与因变量相关性（肯德尔）
        # 拼接签列
        df = X_train.copy(deep=True)
        df[self.label_name] = y_train
        # xy_correlation = df.corr('kendall')
        # xy_correlation = xy_correlation[self.label_name]
        # # 组织为dataframe
        # xy_correlation_df = pd.DataFrame({'features': xy_correlation.index, 'y_correlation': xy_correlation})
        # xy_correlation_df = xy_correlation_df.reset_index(drop=True)
        #
        # # 1.1 首先过滤掉自变量与因变量相关性小的特征
        # filtered_xy_correlation_df = xy_correlation_df[
        #     abs(xy_correlation_df['y_correlation']) > self.xy_correlation_threshold]

        # 2.1 在自变量相关性表中排除掉与因变量小的特征
        # y_correlation_columns = filtered_xy_correlation_df['features'].tolist()
        # xx_correlation_df = xx_correlation_df[xx_correlation_df['feature_base'].isin(y_correlation_columns)]
        # xx_correlation_df = xx_correlation_df[xx_correlation_df['feature_compare'].isin(y_correlation_columns)]

        iv_list = []
        col_list = []
        for col in X_train.columns.tolist():
            woe_iv_dict = _calc_woe_iv(df, col, self.label_name)
            col_list.append(col)
            iv_list.append(woe_iv_dict["IV"])

        filtered_xy_correlation_df = pd.DataFrame({'features': col_list, 'y_correlation': iv_list})

        # 2.2 筛选出自变量间高相关度特征，准备删除与因变量相关性低的一个
        xx_high_correlation_df = xx_correlation_df[
            abs(xx_correlation_df['correlation']) >= self.xx_correlation_threshold]

        # 2.3 在高自变量相关性表中，关联拼接因变量相关性值信息
        xx_high_correlation_df = xx_high_correlation_df.merge(filtered_xy_correlation_df, left_on='feature_base',
                                                              right_on='features', how='left')
        xx_high_correlation_df.rename(index=str, columns={'y_correlation': 'base_y_correlation'}, inplace=True)
        xx_high_correlation_df = xx_high_correlation_df.merge(filtered_xy_correlation_df, left_on='feature_compare',
                                                              right_on='features', how='left')
        xx_high_correlation_df.rename(index=str, columns={'y_correlation': 'compare_y_correlation'}, inplace=True)

        # 2.4 找出与因变量相关性小的特征
        less_correlation_list = []
        for index, row in xx_high_correlation_df.iterrows():
            if row['base_y_correlation'] >= row['compare_y_correlation']:
                less_correlation_list.append(row['feature_compare'])
            else:
                less_correlation_list.append(row['feature_base'])

        # 3.1 删除与因变量相关性小的特征，得到相关性筛选后的特征列表
        filtered_features_set = set(filtered_xy_correlation_df['features'].tolist()) - set(less_correlation_list)
        # filtered_features_set.remove(self.label_name)

        # 4.1 判断保留最大特征数。超出最大特征数，截取前N个与因变量相关性高的特征
        if len(filtered_features_set) > self.max_column_number_threshold - len(set(self.keep_columns_list)):
            # 因变量相关性排序
            filtered_xy_correlation_df = filtered_xy_correlation_df[
                filtered_xy_correlation_df['features'].isin(filtered_features_set)]
            filtered_xy_correlation_df.sort_values(['y_correlation'], ascending=False, inplace=True)
            # 截取特征
            keep_feature_number = len(filtered_features_set) - (
                    self.max_column_number_threshold - len(set(self.keep_columns_list)))
            filtered_features_set = set(filtered_xy_correlation_df['features'].tolist()[:keep_feature_number])

        # 5.1 拼接保留特征
        feature_set = filtered_features_set | set(self.keep_columns_list)

        # 6.1 按原有顺序排序特征
        feature_list = []
        for column in df.columns:
            if column in feature_set:
                feature_list.append(column)

        self.model = feature_list

        return self


class FeatureSelectionByMonotonicityRule(RuleMixin,
                                         BaseEstimator):
    """
    功能：整理评分卡应用的特征单调性筛选输入数据
    输入：整理的分箱编码数据集 df
    控制：保留特征列表         keep_columns_list
    输出：过滤后特征数据集
    """
    keep_columns_list: List[str] = []

    filtered_df: pd.DataFrame = None

    # def _judge_monotonicity(self, encoding_group, encoding):
    #     # 剔出“空值分箱”，空分箱不参与单调性判断
    #     judge_df = encoding_group.copy(deep=True)
    #     if judge_df['binning'].tolist()[-1] in [np.nan, 'nan', 'NaN']:
    #         judge_df = judge_df.iloc[:-1, :]
    #     # 分箱数大于等于 4 时执行单调性判断
    #     if len(judge_df.loc[:, encoding]) >= 4:
    #         # 用类似于 2 阶导数的方法判断单调性
    #         encoding_list = judge_df.loc[:, encoding].values.tolist()
    #
    #         # 计数类似于一阶导数的数值列表，递增与平行值为1，递减为-1
    #         derivative_list = []
    #         for i in range(len(encoding_list) - 1):
    #             if encoding_list[i + 1] >= encoding_list[i]:
    #                 derivative_list.append(1)
    #             else:
    #                 derivative_list.append(-1)
    #         # 计算类似于二阶导数的数值列表，“一阶导数”不变时为0，变化时（出现单调性变化）为1
    #         second_derivative_list = []
    #         for j in range(len(derivative_list) - 1):
    #             if derivative_list[j + 1] - derivative_list[j] == 0:
    #                 second_derivative_list.append(0)
    #             else:
    #                 second_derivative_list.append(1)
    #
    #         # 判断二阶导数，点调性变化最多只能出现一次，否则剔出此特征
    #         # 单调性过滤仅保留 递增，递减 或者 出现过一次单调性变化的特征
    #         if second_derivative_list.count(0) >= len(encoding_list) - 3:
    #             judgement = 1
    #         else:
    #             judgement = 0
    #         encoding_group['judgement'] = judgement
    #     # 分箱数为 2 或 3 时，默认保留此特征
    #     elif (len(judge_df.loc[:, encoding]) == 2) or (len(judge_df.loc[:, encoding]) == 3):
    #         encoding_group['judgement'] = 1
    #     # 分箱数为 1 的情况，剔出此特征
    #     else:
    #         encoding_group['judgement'] = 0
    #     return encoding_group

    def _filter_data(self, df):
        """
        保留具有单调性的变量
        :param df: 带单调性判断的编码规则
        :return: 具有单调性的编码规则
        """
        df = df.reset_index(drop=True)
        # 构造原始特征，分箱方法，分箱数列，用于特征筛选
        for i in range(len(df)):
            feature = df.loc[i, 'column_name']
            df.loc[i, 'original_name'] = '_'.join(str(feature).split('_')[:-1])
            df.loc[i, 'binning_method'] = 'ChiMerge' if 'ChiMerge' in feature.split('_')[-1] else 'DecisionTree'
            df.loc[i, 'binning_number'] = feature.split('_')[-1][8:] if 'ChiMerge' in feature else feature.split('_')[
                                                                                                       -1][12:]

        # 获取待进一步特征过滤的数据集
        pass_df = df.loc[df.loc[:, 'judgement'] == 1, :]
        keep_df = df.loc[
                  df.loc[:, 'column_name'].apply(lambda x: '_'.join(x.split('_')[:-1])).isin(self.keep_columns_list), :]
        work_df = pd.concat([pass_df, keep_df], axis=0)

        # 对于同一特征，同一分箱，同一编码情况下，删除分箱数较小的一组分箱
        work_df.sort_values(by=['original_name', 'binning_method', 'encoding_method', 'binning_number'], inplace=True)
        work_df.drop_duplicates(subset=['original_name', 'binning_method', 'encoding_method'], keep='last',
                                inplace=True)
        work_df = work_df.loc[:, ['original_name', 'binning_method', 'encoding_method', 'binning_number']]
        # 过滤各特征
        filtered_df = df.merge(work_df)
        filtered_df.drop(['original_name', 'binning_method', 'binning_number', 'judgement'], axis=1, inplace=True)

        return filtered_df

    def fit(self, df):
        """
        单调性判断服务
        :param df: 单调性判断输入数据
        :return: 具有单调性的编码规则
        """
        binning_encoding_df = df.reset_index(drop=True)
        # 按原顺序去重，用于按照列名和编码方法分组取出数据块
        binning_column_list = []
        encoding_method_list = []
        for item in binning_encoding_df['column_name'].tolist():
            if item not in binning_column_list:
                binning_column_list.append(item)
        for item in binning_encoding_df['encoding_method']:
            if item not in encoding_method_list:
                encoding_method_list.append(item)
        # 单调性判断
        judgement_df = pd.DataFrame()
        for column in binning_column_list:
            for method in encoding_method_list:
                # 按组判断单调性
                encoding_group = binning_encoding_df.loc[(binning_encoding_df['column_name'] == column) & (
                        binning_encoding_df['encoding_method'] == method), :]
                if len(encoding_group) > 0:
                    encoding_group = _judge_monotonicity(encoding_group, 'encoding')
                    judgement_df = pd.concat([judgement_df, encoding_group], axis=0)
        # 过滤特征列表
        filtered_df = self._filter_data(judgement_df.reset_index(drop=True))
        self.filtered_df = filtered_df.copy(deep=True)
        filtered_df['filtered_column'] = filtered_df['column_name'] + '_' + filtered_df['encoding_method']
        filtered_df.drop_duplicates(subset=['filtered_column'], keep='first', inplace=True)
        self.model = filtered_df['filtered_column'].tolist()

        return self


class FeatureSelectionByVIFRule(RuleMixin,
                                BaseEstimator):
    """
    功能： 每轮循环中计算各个变量的VIF，并删除VIF>threshold 的变量(通常阈值定义为10)
    输入： 待处理数据集 X
    控制： 阈值，默认10.0
    输出： 特征筛选后的数据集
    """
    # 阈值
    VIF_thres: confloat(ge=0, le=1000) = 10.0
    # 保留带着VIF的数据表
    VIF_filter: pd.DataFrame = None

    def fit(self, X):
        col = list(range(X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(X.iloc[:, col].values, ix)
                   for ix in range(X.iloc[:, col].shape[1])]
            vif1 = pd.DataFrame()
            vif1["VAR_NAME"] = list(X.columns[col])
            vif1["VIF_Factor"] = vif
            maxvif = max(vif)
            maxix = vif.index(maxvif)
            if maxvif > self.VIF_thres:
                del col[maxix]
                dropped = True
        self.VIF_filter = vif1
        self.model = vif1["VAR_NAME"].tolist()
        return self


class FeatureSelectionByIVRule(RuleMixin,
                               BaseEstimator):
    """
    功能： 计算每个离散变量的IV值，删除IV值小于0.02的变量
    输入： 待处理数据集 X_train
          标签数据    y_train
    控制： 阈值，默认0.02
    输出： 特征筛选后的数据集
    """
    # 阈值
    IV_thres: confloat(ge=0, le=1) = 0.02
    # 保留带着IV值的数据表
    IV_filter: pd.DataFrame = None

    def fit(self, X_train, y_train):
        df = pd.concat([X_train, y_train], axis=1)
        WOE_COL = X_train.columns.tolist()
        target = 'label'
        WOE = pd.DataFrame()
        IV = []
        for col in WOE_COL:
            WOE_IV = _calc_woe_iv(df, col, target)
            TEMP_WOE = pd.DataFrame({'COL_CLASS': list(WOE_IV['WOE'].keys()), 'WOE': list(WOE_IV['WOE'].values())},
                                    columns=['COL_CLASS', 'WOE'])
            TEMP_WOE['VAR_NAME'] = col
            WOE = pd.concat([WOE, TEMP_WOE], axis=0)
            IV.append(WOE_IV["IV"])
        IV = pd.DataFrame({'VAR_NAME': WOE_COL, 'IV_VALUE': IV}, columns=['VAR_NAME', 'IV_VALUE'])
        self.IV_filter = IV.loc[IV.loc[:, 'IV_VALUE'] >= self.IV_thres, :]

        self.model = self.IV_filter['VAR_NAME'].tolist()
        return self


class FeatureSelectionByPValueAuc(RuleMixin,
                                  BaseEstimator):
    """
    功能：基于P值和AUS值的逻辑回归特征筛选
    输入：训练集自变量表 X_train
          训练集标签 y_train
    控制：指明标签列: label_column
          逻辑回归算法: is_sklearn_LR
          基础分: basic_score
          翻倍分: pdo
          P0: po
          P值筛选阈值：p_value_threshold
          最小特征组合数：min_column_number
    输出：特征筛选后的数据集
    """
    label_column: str = 'label'
    is_sklearn_LR: bool = False
    basic_score: int = 600
    pdo: int = 20
    p0: confloat(ge=0, le=1) = 0.05
    p_value_threshold: confloat(ge=0, le=1) = 0.05
    min_column_number: int = 4

    aim_auc: float = 0.0
    main_test_auc_dict: dict = {}
    X_train: DataFrame = None
    X_test: DataFrame = None
    y_train: Series = None
    y_test: Series = None

    def loop_columns(self, column_list):
        # 递归停止条件
        if len(column_list) <= self.min_column_number:
            return

        test_auc_dict = {}

        finded_column_list = []
        loop_len = len(column_list) + 1
        for i in range(loop_len):
            # 通过P值找特征列表
            cols = column_list.copy()
            # 依次有放回的删除一个特征
            if i - 1 >= 0:
                cols.pop(i - 1)
            scorecard = ScorecardModel(label_column=self.label_column, is_sklearn_LR=self.is_sklearn_LR,
                                       basic_score=self.basic_score, pdo=self.pdo, p0=self.p0,
                                       p_value_threshold=self.p_value_threshold)
            scorecard.fit(self.X_train[cols], self.y_train)
            filtered_col_list = scorecard.p_value_df['columns'].tolist()

            filtered_col_set = set(filtered_col_list)
            if filtered_col_set in finded_column_list:
                # 如果是已有组合，就pass
                pass
            else:
                finded_column_list.append(filtered_col_set)
                # 计算auc
                scorecard.model['binning'] = scorecard.model['encoding']
                y_train_predict = scorecard.predict(self.X_train[filtered_col_list])
                y_test_predict = scorecard.predict(self.X_test[filtered_col_list])
                train_auc = 1 - roc_auc_score(self.y_train, y_train_predict)
                test_auc = 1 - roc_auc_score(self.y_test, y_test_predict)
                # 记录 结果
                col_dict = {'train_auc': train_auc, 'test_auc': test_auc,
                            'filtered_columns': filtered_col_list, 'input_columns': column_list}

                print(json.dumps(col_dict))
                # 保存结果，后续筛选auc最大值
                test_auc_dict[test_auc] = col_dict
        # 汇集auc，用于最后的判断
        self.main_test_auc_dict.update(test_auc_dict)
        # 递归，列数一定要减少
        test_max_auc = max(test_auc_dict.keys())
        test_column_list = test_auc_dict[test_max_auc]['filtered_columns']
        if len(test_column_list) == len(column_list):
            test_auc_dict.pop(test_max_auc)
            test_max_auc = max(test_auc_dict.keys())
            test_column_list = test_auc_dict[test_max_auc]['filtered_columns']

        self.loop_columns(test_column_list)

    def fit(self, x, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        column_list = self.X_train.columns.tolist()

        self.loop_columns(column_list)

        # 转换为dataframe
        df_dict = {}
        train_auc_list = []
        test_auc_list = []
        filtered_columns_list = []
        input_columns_list = []
        for _, values in self.main_test_auc_dict.items():
            train_auc_list.append(values['train_auc'])
            test_auc_list.append(values['test_auc'])
            filtered_columns_list.append(values['filtered_columns'])
            input_columns_list.append(values['input_columns'])
        df_dict['train_auc'] = train_auc_list
        df_dict['test_auc'] = test_auc_list
        df_dict['filtered_columns'] = filtered_columns_list
        df_dict['input_columns'] = input_columns_list
        res_df = pd.DataFrame(df_dict)
        self.aim_auc = max(self.main_test_auc_dict.keys())
        self.model = res_df
        return self

    def transform(self, df):
        filtered_columns = self.model.loc[self.model['test_auc'] == self.aim_auc, 'filtered_columns'].tolist()
        if len(filtered_columns) == 0:
            raise ValueError('Can not find giving AUC value: {0}, please check it.\n'
                             '找不到给定AUC值对应的记录，AUC: {0}，请检查。'.format(str(self.aim_auc)))
        else:
            filtered_columns = filtered_columns[0]
        return df[filtered_columns]


class FeatureSelectionandBinning(RuleMixin,
                                   BaseEstimator):
    """
    功能：利用贪心的思想将多分箱截取成指定数量的分箱
    输入：数据集 df
    控制： 粗糙分箱数 coarse_bins_number
          目标分箱数 target_bins_number
    输出： 过滤的数据集 df
    """
    coarse_bins_number: int = 20
    target_bins_number: int = 6

    def fit(self, X_train, y_train):
        greedy_model_dict = dict()

        for colName in X_train.columns:
            # 声明字典value为列表
            greedy_model_dict[colName] = []

            # 获取非空值的索引列表
            # self.index_isnull = X_train[X_train[colName].isnull()].index.tolist()
            # X_train_column_true = X_train[colName].drop(index=self.index_isnull)
            # y_train_true = y_train.drop(index=self.index_isnull)
            # 级联features和label
            # df = pd.concat([X_train_column_true, y_train_true], axis=1)
            df = pd.concat([X_train[colName], y_train], axis=1)
            # 根据粗糙分箱数等频分箱 会出现bin边界重复的问题 所以这里不用
            # result, bins_array = pd.qcut(df[colName], q=self.coarse_bins_number, retbins=True)
            # 根据粗糙分箱数等距分箱
            result, bins_array = pd.cut(df[colName], bins=self.coarse_bins_number, retbins=True)
            bins_list = bins_array.tolist()
            # 将首尾独立 因为pd.cut的使用需要首尾 但是查找切分点时不需要
            start = bins_list[0]
            end = bins_list[-1]
            bins = bins_list[1:-1]

            # 初始解集为空
            best_subbins = []
            # 初始maxv值设置为0
            maxv = 0
            # 循环停止的标签
            stop_flag = False

            while len(best_subbins) < self.target_bins_number and (not stop_flag):
                find_bin_edge = None
                for bin_edge in bins:
                    best_subbins.append(bin_edge)
                    best_subbins.sort()
                    # 计算当前分箱的woe和iv值
                    bin_col_name = colName + '_bin'
                    df[bin_col_name] = pd.cut(df[colName], [start] + best_subbins + [end])
                    woe_iv_dict = _calc_woe_iv(df, col=bin_col_name, target='label')
                    woe_dict = woe_iv_dict.get("WOE")
                    iv_tmp = woe_iv_dict.get("IV")
                    # 判断单调性
                    df_woe = pd.DataFrame([woe_dict]).T
                    df_woe = df_woe.reset_index(drop=False)
                    df_woe.rename(columns={'index': 'bin_edge', 0: "woe_value"}, inplace=True)
                    encoding_group = _judge_monotonicity(df_woe, 'woe_value')

                    if encoding_group.loc[0, 'judgement'] == 1:
                        if iv_tmp > maxv:
                            maxv = iv_tmp
                            find_bin_edge = bin_edge
                    best_subbins.remove(bin_edge)

                if find_bin_edge:
                    best_subbins.append(find_bin_edge)
                    best_subbins.sort()
                    bins.remove(find_bin_edge)
                else:
                    stop_flag = True

            if len(best_subbins) != self.target_bins_number:
                best_subbins = []

            # 将subbins保存到对应字典
            greedy_model_dict[colName].extend(best_subbins)

        self.model = greedy_model_dict
        return self

    def transform(self, X_train):
        """
        数值列自动应用选择分箱服务
        ：para X_train: 待选择分箱的数据
        ：return“: 选择分箱后的数据
        """
        # 获取选择分箱字典
        greedy_model = self.model

        for colName in X_train.columns:
            best_subbins = greedy_model.get(colName)
            print(type(best_subbins))
            if best_subbins == []:
                X_train = X_train.drop(columns=colName, axis=1)
            else:
                start = np.nanmin(X_train[colName].values) - 1
                end = np.nanmax(X_train[colName].values)
                bins_add = [start] + best_subbins + [end]
                X_train[colName] = pd.cut(X_train[colName], bins=bins_add)
                X_train[colName] = X_train[colName].astype('str')
                '''
                # 修改最小分箱左边界值
                min_interval_str = '({}, {}]'.format(bins_add[0], bins_add[1])
                index_min_list = X_train[X_train[colName] == min_interval_str].index.tolist()
                X_train[colName].iloc[index_min_list] = '(-inf, {}]'.format(bins_add[1])
                # 修改最大分箱右边界值
                max_interval_str = '({}, {}]'.format(bins_add[-2], bins_add[-1])
                index_max_list = X_train[X_train[colName] == max_interval_str].index.tolist()
                X_train[colName].iloc[index_max_list] = '({}, inf]'.format(bins_add[-2])
                '''
                X_train.rename(columns={colName:colName+'_DecisionTree6'},inplace=True)

        return X_train
