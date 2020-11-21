#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/1 14:50
# filename: _processors_test
# software: PyCharm

import os
import unittest
import pandas as pd

from pandas import DataFrame
from pandas.testing import assert_series_equal
from pandas.testing import assert_frame_equal

from ..modules.util import DataTypeEnum
from ..modules.data_manipulation import ColumnConcatingHandler
from ..modules.data_manipulation import ColumnTypeConvertingHandler
from ..modules.data_manipulation import ColumnTypeFilteringHandler
from ..modules.data_manipulation import DropAffixHandler
from ..modules.data_manipulation import RowPartitioningHandler
from ..modules.data_manipulation import RowCleaningHandler
from ..modules.data_manipulation import DataframesFeatureFilteringHandlar
from ..modules.data_manipulation import OutlierHandlingRule
from ..modules.data_manipulation import ToDataframeHandler
from ..modules.data_manipulation import UsingFeaturetoolsHandler
from ..modules.data_manipulation import RowConcatingHandler
from ..modules.data_io import CSVReadingHandler
from ..modules.feature_engineering import TwoDatetimeColumnsSubtractHandler
from ..modules.feature_engineering import DatetimeColumnDecomposeHandler
from ..modules.feature_engineering import NumberColumnsCalculatingHandler
from ..modules.feature_engineering import CategoryColumnsComposeHandler

"""
参考资料：https://docs.python.org/zh-cn/3/library/unittest.html
"""


class TestHandlers(unittest.TestCase):
    # 测试用例前置条件
    def setUp(self):
        bath_path = os.getcwd()
        self._df = pd.read_csv(os.path.join(bath_path, r'mldesigntoolkit\test\test_data\small_data_set.csv'))
        self._df1 = pd.read_csv(os.path.join(bath_path, r'mldesigntoolkit\test\test_data\test1.csv'))
        self._df2 = pd.read_csv(os.path.join(bath_path, r'mldesigntoolkit\test\test_data\test2.csv'))
        self._df3 = pd.read_csv(os.path.join(bath_path, r'mldesigntoolkit\test\test_data\test3.csv'))
        self._df_left = pd.read_csv(
            os.path.join(bath_path, r'mldesigntoolkit\test\test_data\small_data_set_left.csv'))
        self._df_right = pd.read_csv(
            os.path.join(bath_path, r'mldesigntoolkit\test\test_data\small_data_set_right.csv'))
        self._df_two_datetime = pd.read_csv(
            os.path.join(bath_path, r'mldesigntoolkit\test\test_data\two_datetime_df.csv'),
            parse_dates=['latter_date', 'former_date'])
        self._df_two_datetime_subtract_true = pd.read_csv(
            os.path.join(bath_path, r'mldesigntoolkit\test\test_data\two_datetime_column_subtract.csv'))
        self._df_datetime = pd.read_csv(os.path.join(bath_path, r'mldesigntoolkit\test\test_data\datetime_df.csv'),
                                        parse_dates=['date_examples1', 'date_examples2'])
        self._df_datetime_decompose_true = pd.read_csv(
            os.path.join(bath_path, r'mldesigntoolkit\test\test_data\datetime_decompose.csv'))
        self._df_number = pd.read_csv(os.path.join(bath_path, r'mldesigntoolkit\test\test_data\number_calculate.csv'))
        self._df_number_calculate_true = pd.read_csv(
            os.path.join(bath_path, r'mldesigntoolkit\test\test_data\number_calculate_true.csv'))
        self._df_category = pd.read_csv(os.path.join(bath_path, r'mldesigntoolkit\test\test_data\category_compose.csv'))
        self._df_category_compose_true = pd.read_csv(
            os.path.join(bath_path, r'mldesigntoolkit\test\test_data\category_compose_true.csv'))
        # featuretools 测试数据
        self._transaction_data = pd.read_csv(
            os.path.join(bath_path, r'mldesigntoolkit\test\test_data\transaction_data.csv'), nrows=10000)
        self._transaction_data_cutoff_times = pd.read_csv(
            os.path.join(bath_path, r'mldesigntoolkit\test\test_data\transaction_data_cutoff_times.csv'))

    # 方法名必须以 test_ 开头
    def test_ColumnTypeConvertingHandler(self):
        # 测试 ColumnTypeConvertingHandler 组件
        _handler_object = ColumnTypeConvertingHandler()
        result_df = _handler_object.process(self._df.copy(deep=True))
        new_index = self._df.columns
        data_list = ['category'] * len(new_index)
        result_series = pd.Series(data_list, index=new_index)
        print()
        print(self._df.dtypes)
        print()
        print(result_series)
        assert_series_equal(result_df.dtypes, result_series)

    def test_ColumnTypeFilteringHandler(self):
        # 测试 ColumnTypeFilteringHandler 组件
        _handler_object = ColumnTypeFilteringHandler(filtering_type='object')
        result_df = _handler_object.process(self._df)
        compare_df = self._df.select_dtypes(include='object')
        assert_frame_equal(result_df, compare_df)
        print(result_df.columns)

    def test_DropAffixHandler(self):
        # 测试 DropAffixHandler 组件
        _handler_object = DropAffixHandler()
        result_df = _handler_object.process(self._df.copy(deep=True))
        compare_df = self._df.copy(deep=True)
        compare_df.columns = ['_'.join(str(x).split('_')[:-1]) if len(str(x).split('_')) > 1 else x
                              for x in compare_df.columns]
        print()
        print(result_df.columns)
        print()
        print(compare_df.columns)
        assert_frame_equal(result_df, compare_df)

    def test_RowPartitioningHandler(self):
        # 测试 RowPartitioningHandler 组件
        _handler_object = RowPartitioningHandler()
        train_df, test_df = _handler_object.process(self._df)
        self.assertEqual(self._df.shape[0], (train_df.shape[0] + test_df.shape[0]))
        self.assertEqual(round(train_df.shape[0] / self._df.shape[0], 1), 0.7)

    def test_RowCleaningHandler(self):
        # 测试 RowCleaningHandler 组件
        _handler_object = RowCleaningHandler()
        result_df = _handler_object.process(self._df)
        record_length = result_df.shape[1]
        result_df['_number_missing'] = result_df.isnull().sum(axis=1)
        flag = (result_df['_number_missing'] / record_length) >= 0.8
        print()
        print(result_df[flag])
        self.assertEqual(result_df[flag].shape[0], 0)

    def test_ToDataframeHandler(self):
        _fit_object = OutlierHandlingRule(
            continuous_columns_list=['supervise_item_implement_code', 'administrative_cp_na',
                                     'administrative_cp_ce_type', 'area_number', 'check_form'],
            discrete_columns_list=['check_type', 'check_mode', 'check_result', 'sourceno', 'src_rec_batch_no'])
        _model_object = _fit_object.fit(self._df.copy(deep=True))
        print()
        print(_model_object.model)
        self.assertIsInstance(_model_object.model, DataFrame)
        # 测试 ToDataframeHandler 组件
        _handler_object = ToDataframeHandler()
        result_df = _handler_object.process(_model_object)
        print()
        print('To DataFrame content:')
        print(result_df)
        self.assertIsInstance(result_df, DataFrame)

    def test_DataframesFeatureFilteringHandlar(self):
        _handler_object = DataframesFeatureFilteringHandlar()
        result_df = _handler_object.process(self._df1, self._df2)
        self.assertEqual(self._df1.shape[1], result_df.shape[1])

    #     result_df_1 = _handler_object.process(self._df1, self._df3)
    #     self.assertEqual(self._df1.shape[1], result_df_1.shape[1])

    def test_CSVReadingHandler(self):
        bath_path = os.getcwd()
        test_file_path = os.path.join(bath_path, r'mldesigntoolkit\test\test_data\small_data_set.csv')
        _handler_object = CSVReadingHandler(file_path=test_file_path)
        result_df = _handler_object.process()
        assert_frame_equal(result_df, self._df)

    def test_ColumnConcatingHandler(self):
        _handler_object = ColumnConcatingHandler(join_column_list=['cd_id'], join_method='left')
        result_df = _handler_object.process(self._df_left, self._df_right)
        df = self._df_left.merge(self._df_right, on=['cd_id'], how='left', suffixes=['_left', '_right'])
        assert_frame_equal(result_df, df)

    def test_RowConcatingHandler(self):
        _handler_object = RowConcatingHandler()
        result_df = _handler_object.process(self._df1, self._df3)
        self.assertEqual(self._df1.shape[1], result_df.shape[1])

    def test_TwoDatetimeColumnsSubtractHandler(self):
        _handler_object = TwoDatetimeColumnsSubtractHandler(datetime_columns_list=['latter_date', 'former_date'],
                                                            datetime_features_list=['age', 'year',
                                                                                    'month', 'quarter',
                                                                                    'day', 'week'])
        result_df = _handler_object.process(self._df_two_datetime)
        assert_frame_equal(result_df, self._df_two_datetime_subtract_true)

    def test_DatetimeColumnDecomposeHandler(self):
        _handler_object = DatetimeColumnDecomposeHandler(datetime_columns_list=['date_examples1', 'date_examples2'],
                                                         datetime_features_list=['year', 'week', 'month', 'quarter',
                                                                                 'halfyear', 'halfmonth', 'dayofweek',
                                                                                 'isholiday', 'isworkday', 'isweekend'])
        result_df = _handler_object.process(self._df_datetime)
        assert_frame_equal(result_df, self._df_datetime_decompose_true)

    def test_NumberColumnsCalculatingHandler(self):
        _handler_object = NumberColumnsCalculatingHandler(number_columns_list=['distance', 'time'],
                                                          calculating_type='divide')
        result_df = _handler_object.process(self._df_number)
        series_name = 'distance_divide_time'
        assert_series_equal(result_df[series_name], self._df_number_calculate_true[series_name])

    def test_CategoryColumnsComposeHandler(self):
        _handler_object = CategoryColumnsComposeHandler(category_columns_list=['large', 'small'])
        result_df = _handler_object.process(self._df_category)
        assert_frame_equal(result_df, self._df_category_compose_true)

    def test_UsingFeaturetoolsHandler(self):
        self._transaction_data_cutoff_times.columns = ['loan_id', 'time']
        self._transaction_data_cutoff_times['time'] = pd.to_datetime(self._transaction_data_cutoff_times['time'],
                                                                     format="%Y-%m-%d", errors='coerce')
        _one_month_ago_cutoff_time = self._transaction_data_cutoff_times.copy(deep=True)

        import dateutil
        _one_month_ago_cutoff_time['time'] = _one_month_ago_cutoff_time['time'].apply(lambda x: x-dateutil.relativedelta.relativedelta(months=1))
        _handler_object = UsingFeaturetoolsHandler(entity_id='loan_id',
                                                   numeric_cols=['amount', 'total_price', 'unit_price'],
                                                   category_cols=['amount_binning', 'unit_price_binning'],
                                                   time_index='order_time',
                                                   cutoff_time_in_index=True,
                                                   cutoff_time_dict={'current': self._transaction_data_cutoff_times,
                                                                     'one_month_ago': _one_month_ago_cutoff_time},
                                                   window_list=['3 month', '6 month'])

        result_df = _handler_object.process(self._transaction_data[['loan_id', 'order_time', 'amount', 'total_price',
                                                                    'unit_price', 'order_time_isweekend',
                                                                    'amount_binning',
                                                                    'unit_price_binning', 'total_price_binning']])
        print()
        print(len(result_df.dtypes))
        print(result_df.dtypes)
        print(result_df.head(10))
        self.assertIsInstance(result_df, DataFrame)

    def tearDown(self):
        # 测试用例回收方法
        pass

    def test_zqz(self):
        # 测试os
        bath_path = os.getcwd()
        print(bath_path +'//123')

    def test_zd_csv(self):



if __name__ == '__main__':
    unittest.main()
