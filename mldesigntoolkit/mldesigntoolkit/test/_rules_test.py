#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/2 16:54
# filename: _rules_test
# software: PyCharm

import os
import unittest
from datetime import datetime

import pandas as pd
import numpy as np
from pandas import DataFrame

from ..modules.data_manipulation import ScorecardModelUpdateHandler
from ..modules.modeling import ScorecardModel
from ..modules.data_manipulation import ColumnCleaningRule
from ..modules.data_manipulation import MissingValuesFillingRule
from ..modules.data_manipulation import OutlierHandlingRule
from ..modules.data_manipulation import HighAndLowCardinalitySplittingRule
from ..modules.feature_engineering import ColumnEncodingRule
from ..modules.feature_engineering import SupervisedColumnEncodingRule
from ..modules.feature_engineering import FeatureSelectionByTreeAlgorithmRule
from ..modules.feature_engineering import FeatureSelectionByCorrelationRule
from ..modules.feature_engineering import FeatureSelectionByMonotonicityRule
from ..modules.feature_engineering import ColumnBinningRule
from ..modules.feature_engineering import SupervisedColumnBinningRule
from ..modules.data_manipulation import MonotonicitySelectionRuleDataHandler
from ..modules.feature_engineering import FeatureSelectionByVIFRule
from ..modules.feature_engineering import FeatureSelectionByIVRule
from ..modules.feature_engineering import FeatureSelectionandBinning

"""
参考资料：https://docs.python.org/zh-cn/3/library/unittest.html
"""

# 显示所有列
pd.set_option('display.max_columns', None)

# 显示所有行
pd.set_option('display.max_rows', None)

# 控制台不回行
pd.set_option('display.width', 1000)

# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 1000)


class TestRules(unittest.TestCase):

    def setUp(self):
        # 测试用例前置条件
        bath_path = os.getcwd()
        self._df = pd.read_csv(os.path.join(bath_path, r'mldesigntoolkit\test\test_data\small_data_set.csv'))
        self._df_number = pd.read_csv(os.path.join(bath_path, r'mldesigntoolkit\test\test_data\adultData_drop_id.csv')
                                      , nrows=100)

    def test_ColumnCleaningRule(self):
        # 测试 ColumnCleaningRule 组件 fit 功能
        _fit_object = ColumnCleaningRule()
        _model_object = _fit_object.fit(self._df.copy(deep=True))
        self.assertIsInstance(_model_object, ColumnCleaningRule)
        self.assertIsInstance(_model_object.model, list)
        # 测试 ColumnCleaningRule 组件 transform 功能
        result_df = _model_object.transform(self._df.copy(deep=True))
        print()
        print(result_df.columns.tolist())
        print(type(result_df.columns))
        print(_model_object.model)
        self.assertSequenceEqual(result_df.columns.tolist(), _model_object.model)

    def test_MissingValuesFillingRule(self):
        _middle_fit_obj = ColumnCleaningRule()
        _middle_model_obj = _middle_fit_obj.fit(self._df.copy(deep=True))
        cleand_df = _middle_model_obj.transform(self._df.copy(deep=True))

        # 测试 MissingValuesFillingRule 组建 fit 功能
        _fit_object = MissingValuesFillingRule(filling_continuous_method='mean', filling_discrete_method='mode')
        _model_object = _fit_object.fit(cleand_df)
        print()
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, dict)
        # 测试 MissingValuesFillingRule 组建 transform 功能
        result_df = _model_object.transform(cleand_df)
        self.assertFalse(any(result_df.isnull().any()))

    def test_OutlierHandlingRule(self):
        # 测试 OutlierHandlingRule 组建 fit 功能
        _fit_object = OutlierHandlingRule(
            continuous_columns_list=['supervise_item_implement_code', 'administrative_cp_na',
                                     'administrative_cp_ce_type', 'area_number', 'check_form'],
            discrete_columns_list=['check_type', 'check_mode', 'check_result', 'sourceno', 'src_rec_batch_no'])
        _model_object = _fit_object.fit(self._df.copy(deep=True))
        print()
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, DataFrame)
        # 测试 OutlierHandlingRule 组建 transform 功能
        result_df = _model_object.transform(self._df.copy(deep=True))
        self.assertIsInstance(result_df, DataFrame)

    def test_HighAndLowCardinalitySplittingRule(self):
        # 测试 HighAndLowCardinalitySplittingRule 组建 fit 功能
        _fit_object = HighAndLowCardinalitySplittingRule()
        _model_object = _fit_object.fit(self._df.copy(deep=True))
        print()
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, dict)
        # 测试 HighAndLowCardinalitySplittingRule 组建 transform 功能
        result_df = _model_object.transform(self._df.copy(deep=True))
        self.assertIsInstance(result_df, DataFrame)
        self.assertSequenceEqual(result_df.columns.tolist(), _model_object.model['high_cardinality_cols'])

    def test_ColumnEncodingRule(self):
        # 测试 ColumnEncodingRule 组建 fit 功能
        _fit_object = ColumnEncodingRule(encoding_methods_list=['onehot', 'count', 'freq'])
        _model_object = _fit_object.fit(self._df[['label', 'implement_institution']])
        print('ModelMixin content:')
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, dict)
        # 测试 ColumnEncodingRule 组建 transform 功能
        result_df = _model_object.transform(self._df[['label', 'implement_institution']])
        print('Result df')
        print(result_df.head())
        self.assertIsInstance(result_df, DataFrame)

    def test_SupervisedColumnEncodingRule(self):
        # 测试 SupervisedColumnEncodingRule 组建 fit 功能
        _fit_object = SupervisedColumnEncodingRule(encoding_methods_list=['catboost', 'target', 'woe'])
        _model_object = _fit_object.fit(self._df[['implement_institution']], self._df['label'])
        print()
        print('ModelMixin content:')
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, dict)
        # 测试 SupervisedColumnEncodingRule 组建 transform 功能
        result_df = _model_object.transform(self._df[['implement_institution']])
        print('Result df')
        print(result_df.head())
        self.assertIsInstance(result_df, DataFrame)

    def test_FeatureSelectionByTreeAlgorithmRule(self):
        # 测试 FeatureSelectionByTreeAlgorithmRule 组建 fit 功能
        _fit_object = FeatureSelectionByTreeAlgorithmRule()
        _model_object = _fit_object.fit(self._df[['supervise_item_implement_code', 'administrative_cp_na',
                                                  'administrative_cp_ce_type', 'area_number', 'check_form',
                                                  'check_type', 'check_mode', 'check_result', 'sourceno',
                                                  'src_rec_batch_no']], self._df['label'])
        print()
        print('ModelMixin content:')
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, list)
        # 测试 FeatureSelectionByTreeAlgorithmRule 组建 transform 功能
        result_df = _model_object.transform(self._df[['supervise_item_implement_code', 'administrative_cp_na',
                                                      'administrative_cp_ce_type', 'area_number', 'check_form',
                                                      'check_type', 'check_mode', 'check_result', 'sourceno',
                                                      'src_rec_batch_no']])
        print('Result df')
        print(result_df.head())
        self.assertIsInstance(result_df, DataFrame)

    def test_FeatureSelectionByCorrelationRule(self):
        # 测试 FeatureSelectionByCorrelationRule 组建 fit 功能
        _fit_object = FeatureSelectionByCorrelationRule(xx_correlation_threshold=0.1, xy_correlation_threshold=0.001)
        _model_object = _fit_object.fit(self._df[['supervise_item_implement_code', 'administrative_cp_na',
                                                  'administrative_cp_ce_type', 'area_number', 'check_form',
                                                  'check_type', 'check_mode', 'check_result', 'sourceno',
                                                  'src_rec_batch_no']], self._df['label'])
        print()
        print('ModelMixin content:')
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, list)
        # 测试 FeatureSelectionByCorrelationRule 组建 transform 功能
        result_df = _model_object.transform(self._df[['supervise_item_implement_code', 'administrative_cp_na',
                                                      'administrative_cp_ce_type', 'area_number', 'check_form',
                                                      'check_type', 'check_mode', 'check_result', 'sourceno',
                                                      'src_rec_batch_no']])
        print('Result df')
        print(result_df.head())
        self.assertIsInstance(result_df, DataFrame)

    def test_ColumnBinningRule(self):
        # 测试 ColumnBinningRule 组建 fit 功能
        _fit_object = ColumnBinningRule()
        _model_object = _fit_object.fit(self._df[['supervise_item_implement_code', 'administrative_cp_na',
                                                  'administrative_cp_ce_type', 'area_number', 'check_form',
                                                  'check_type', 'check_mode', 'check_result', 'sourceno',
                                                  'src_rec_batch_no']])
        print()
        print('1. ModelMixin content:')
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, list)
        # 测试 ColumnBinningRule 组建 transform 功能
        result_df = _model_object.transform(self._df[['supervise_item_implement_code', 'administrative_cp_na',
                                                      'administrative_cp_ce_type', 'area_number', 'check_form',
                                                      'check_type', 'check_mode', 'check_result', 'sourceno',
                                                      'src_rec_batch_no']])
        print('2. Result df')
        print(result_df.head())
        self.assertIsInstance(result_df, DataFrame)

    def test_SupervisedColumnBinningRule(self):
        # 测试 SupervisedColumnBinningRule 组建 fit 功能 卡方分箱
        _fit_object = SupervisedColumnBinningRule(binning_method_list=['ChiMerge'])
        _model_object = _fit_object.fit(self._df[['area_number']], self._df['label'])
        print()
        print('1. ChiMerge ModelMixin content:')
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, list)

        # 测试 SupervisedColumnBinningRule 组建 transform 功能
        result_df = _model_object.transform(self._df[['area_number', 'label']])
        print('2. ChiMerge Result df')
        print(result_df.head())
        self.assertIsInstance(result_df, DataFrame)

        # 测试 SupervisedColumnBinningRule 组建 fit 功能 决策树分箱
        _fit_object = SupervisedColumnBinningRule(binning_method_list=['DecisionTree'])
        _model_object = _fit_object.fit(self._df[['area_number']], self._df['label'])
        print()
        print('1. DecisionTree ModelMixin content:')
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, list)
        # 测试 SupervisedColumnBinningRule 组建 transform 功能
        result_df = _model_object.transform(self._df[['area_number']])
        print('2. DecisionTree Result df')
        print(result_df.head())
        self.assertIsInstance(result_df, DataFrame)

    def test_SupervisedColumnBinningRule_test(self):
        start = datetime.now()
        # 测试 SupervisedColumnBinningRule 组建 fit 功能 卡方分箱
        _fit_object = SupervisedColumnBinningRule(binning_method_list=['DecisionTree'])
        df = self._df_number.copy(deep=True)
        y_train = df.pop('label')
        X_train = df
        _model_object = _fit_object.fit(X_train, y_train)
        print('Running time: ' + str((datetime.now() - start).seconds))
        print('1. ChiMerge ModelMixin content:')
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, list)

        # 测试 SupervisedColumnBinningRule 组建 transform 功能
        result_df = _model_object.transform(X_train)
        print('2. ChiMerge Result df')
        print(result_df)
        self.assertIsInstance(result_df, DataFrame)

    def test_ScorecardModelDataHandler_and_FeatureSelectionByMonotonicityRule(self):
        # 准备数据集
        df = self._df_number[['age', 'fnlwgt', 'capital-gain', 'label']].copy(deep=True)
        y_train = df.pop('label')
        X_train = df
        _fit_object = SupervisedColumnBinningRule(binning_method_list=['ChiMerge'], binning_number_list=[4, 5])
        _model_object = _fit_object.fit(X_train, y_train)
        binning_df = _model_object.transform(X_train)

        _fit_object = SupervisedColumnEncodingRule(encoding_methods_list=['woe', 'target'])
        _model_object = _fit_object.fit(binning_df, y_train)
        encoding_df = _model_object.transform(binning_df)

        # 测试单调性判断数据处理 MonotonicitySelectionRuleDataHandler
        _handler_object = MonotonicitySelectionRuleDataHandler()
        prepare_df = _handler_object.process(binning_df, encoding_df)
        print(prepare_df)
        self.assertIsInstance(prepare_df, DataFrame)

        # 测试单调性判断，过滤组件 FeatureSelectionByMonotonicityRule fit 方法
        _fit_object = FeatureSelectionByMonotonicityRule()
        _model_object = _fit_object.fit(prepare_df)
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, list)

        # 测试单调性判断，过滤组件 FeatureSelectionByMonotonicityRule transform 方法
        result_df = _model_object.transform(encoding_df)
        print(result_df)
        self.assertIsInstance(result_df, DataFrame)

    def test_MonotonicityModelDataHandler_and_ScorecardModel(self):
        # 准备数据集
        df = self._df_number[['age', 'fnlwgt', 'capital-gain', 'label']].copy(deep=True)
        y_train = df.pop('label')
        X_train = df
        _fit_object = SupervisedColumnBinningRule(binning_method_list=['ChiMerge'], binning_number_list=[4])
        _model_object = _fit_object.fit(X_train, y_train)
        binning_df = _model_object.transform(pd.concat([X_train, y_train], axis=1))

        _fit_object = SupervisedColumnEncodingRule(encoding_methods_list=['woe'])
        _model_object = _fit_object.fit(binning_df, y_train)
        encoding_df = _model_object.transform(binning_df)

        _handler_object = MonotonicitySelectionRuleDataHandler()
        prepare_df = _handler_object.process(binning_df, encoding_df)

        # 测试评分卡模型 ScorecardModel fit 方法
        _fit_object = ScorecardModel(is_sklearn_LR=False)
        _scorecard_model_object = _fit_object.fit(encoding_df, y_train)
        print(_scorecard_model_object.model)
        self.assertIsInstance(_scorecard_model_object.model, DataFrame)

        # 测试单调性判断数据处理 MonotonicitySelectionRuleDataHandler
        _handler_object = ScorecardModelUpdateHandler()
        _scorecard_model_object = _handler_object.process(_scorecard_model_object, prepare_df)
        print(_scorecard_model_object.model)
        self.assertIsInstance(_scorecard_model_object.model, DataFrame)

        # 测试评分卡模型 ScorecardModel predict 方法
        result_df = _scorecard_model_object.predict(X_train)
        print(result_df)
        self.assertIsInstance(result_df, DataFrame)

    def test_FeatureSelectionByVIFRule(self):
        # 准备数据集
        df = self._df_number[['age', 'fnlwgt', 'capital-gain', 'label']].copy(deep=True)
        y_train = df.pop('label')
        X_train = df
        _fit_object = FeatureSelectionByVIFRule()
        _model_object = _fit_object.fit(X_train, thres=10.0)
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, DataFrame)

    def test_FeatureSelectionByIVRule(self):
        # 准备数据集
        df = self._df_number[['age', 'fnlwgt', 'capital-gain', 'label']].copy(deep=True)
        y_train = df.pop('label')
        X_train = df
        _fit_object = FeatureSelectionByIVRule()
        _model_object = _fit_object.fit(X_train, y_train)
        print(_model_object.model)
        # self.assertIsInstance(_model_object.model, DataFrame)

    def test_FeatureSelectionandBinning(self):
        # 测试 test_FeatureSelectionandBinning 组件 fit 功能
        _fit_object = FeatureSelectionandBinning(coarse_bins_number=20, target_bins_number=2)
        df = self._df[['area_number', 'label']].copy(deep=True)
        index_list = [1, 2, 3, 4, 8]
        df['area_number'].iloc[index_list] = np.nan
        # df = self._df_number[['age', 'fnlwgt', 'capital-gain', 'label']].copy(deep=True)
        # df = self._df_number[['capital-gain', 'label']].copy(deep=True)
        y_train = df.pop('label')
        X_train = df
        _model_object = _fit_object.fit(X_train, y_train)
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, dict)

        # 测试 test_FeatureSelectionandBinning 组件 transform 功能
        result_df = _model_object.transform(X_train)
        print(result_df)
        self.assertIsInstance(result_df, DataFrame)

    def tearDown(self):
        # 测试用例回收方法
        pass


if __name__ == '__main__':
    unittest.main()
