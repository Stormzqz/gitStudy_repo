#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/1 11:49
# filename: __init__.py
# software: PyCharm

import os
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 230)

from mldesigntoolkit import Invoker

# params = {
#     'control_dict': {
#         'variable_dict': {
#             'params': {'thread_count': -1,
#                        'objective': 'Logloss',
#                        "eval_metric": "AUC",
#                        'verbose': False},
#             'param_spaces': {'learning_rate': ('loguniform', [0.0025, 0.005, 0.01, 0.015, 0.02, 0.025]),
#                              'depth': ('int', [2, 10]),
#                              'colsample_bylevel': ('uniform', [0.1, 1.0]),
#                              'random_strength': ('int', [1, 20]),
#                              'l2_leaf_reg': ('loguniform', [0.1, 10.0]),
#                              'bootstrap_type': ('categorical', ['Bayesian', 'Bernoulli', 'MVS']),
#                              'boosting_type': ('categorical', ['Ordered', 'Plain']), },
#             'n_splits': 5,
#             'n_trials': 20,
#             'early_stopping': 4,
#             'n_models': 5,
#             'random_state': None,
#             'classifier': 'CBDTClassifier'
#         }
#     }
# }
# input_table = pd.read_csv(r'D:\data\train_data.csv')
# y = input_table.pop('label')
# X = input_table
#
# res_list = Invoker.invoke_service('BaggingOptunaClassifierExecutor', [X, y], params)
#
# output_model = res_list[0]
#
#
# params = {}
#
# input_list = [input_table, output_model]
#
# res_list = Invoker.invoke_service('Predictor', input_list, params)
#
# output_table = res_list[0]
#
# params = {
#         'control_dict': {
#
#         }
#     }
#
# res_list = Invoker.invoke_service('ColumnConcatingHandlerExecutor', [y.to_frame(), output_table], params)
#
# output_table = res_list[0]
#
# params = {}
#
# y = output_table.pop('label')
# predict_y = output_table.pop('predict_y')
# input_list = [y, predict_y, output_model]
#
#
# res_list = Invoker.invoke_service('Evaluator', input_list, params)
#
# output_table = res_list[0]
#
#
#
# bath_path = os.getcwd()
# df = pd.read_csv(os.path.join(bath_path, r'test_data\LR_data.csv'))[
#     ['label', 'SOURCE_TYPE_woe', 'HOUSE_PROPERTY_woe', 'RETAIL_TYPE_NEW_woe', 'CUST_SIZE_NEW_woe',
#      'age_ChiMerge5_woe', 'IDENTITY_MONTHS_ChiMerge4_woe', 'YITU_SIMILARITY_ChiMerge6_woe',
#      'TOBACCO_END_MONTH_ChiMerge6_woe', 'register_capital_new_ChiMerge6_woe', 'AMOUNT_7_12_std_ChiMerge6_woe',
#      'AMOUNT_12_std_ChiMerge6_woe', 'TOTAL_PRICE_3_std_ChiMerge6_woe', 'TOTAL_PRICE_4_6_std_ChiMerge6_woe',
#      'TOTAL_PRICE_6_std_ChiMerge5_woe', 'TOTAL_PRICE_7_12_std_ChiMerge6_woe', 'TOTAL_PRICE_12_std_ChiMerge6_woe',
#      'UNIT_PRICE_new_6_std_ChiMerge4_woe',
#      'SUM_AMOUNT_current_3month_subtract_SUM_AMOUNT_up4month_3month_ChiMerge6_woe',
#      'SUM_AMOUNT_current_6month_subtract_SUM_AMOUNT_up7month_6month_ChiMerge6_woe']]
#
# y = df.pop('label')
# X = df
#
# params = {
#     'control_dict': {
#     }
# }
#
# res_list = Invoker.invoke_service('FeatureSelectionByPValueAucExecutor', [X, y], params)
# fs_obj = res_list[0]
# print(fs_obj.model)
# fs_obj.model.to_csv('auc_df.csv')
#
# params = {
#     'control_dict': {
#         'aim_auc': 0.7280107248678804
#     }
# }
#
# input_list = [df, fs_obj]
# column_cleaned_data = Invoker.invoke_service('Transformer', input_list, params)
#
# output_table = column_cleaned_data[0]
# print(output_table.columns)
#
#
# params = {
#     'control_dict': {
#     }
# }
# binning_df = pd.read_csv(r'E:\JupyterNotebookWorkspace\data\binning_df.csv')
# encoding_df = pd.read_csv(r'E:\JupyterNotebookWorkspace\data\encoding_df.csv')
#
# res_list = Invoker.invoke_service('MonotonicitySelectionRuleDataHandlerExecutor', [binning_df, encoding_df], params)
# prepare_df = res_list[0]
# print(prepare_df)
#
# params = {
#     'control_dict': {
#     }
# }

# res_list = Invoker.invoke_service('FeatureSelectionByMonotonicityRuleExecutor', [prepare_df], params)
# select_obj = res_list[0]
# # print(select_obj.model)
# # print()
# # print(select_obj.filtered_df)
#
# params = {
#     'control_dict': {
#         'is_sklearn_LR': False
#     }
# }
#
# # y_train = pd.read_csv(r'D:\data\y_train.csv')
# encoding_df = pd.read_csv(r'D:\data\encoding_data.csv')
# y_train = encoding_df.pop('label')
# scorecard_obj = Invoker.invoke_service('ScorecardModelExecutor', [encoding_df, y_train], params)
# scorecard_obj = scorecard_obj[0]
#
# params = {
#     'control_dict': {
#     }
# }
#
# prepare_df = pd.read_csv(r'D:\data\binning_data.csv')
#
# scorecard_obj = Invoker.invoke_service('ScorecardModelUpdateHandlerExecutor', [scorecard_obj, prepare_df], params)
# scorecard_obj = scorecard_obj[0]
#
# print(scorecard_obj.model)
#
# # params = {}
# #
# # X_train = pd.read_csv(r'D:\data\X_train.csv', encoding='utf-8')
# # input_list = [X_train, scorecard_obj]
# #
# # res_list = Invoker.invoke_service('Predictor', input_list, params)
# #
# # output_table = res_list[0]
# # output_table.to_csv('sc_predict.csv', index=False)
#
# params = {
#     'control_dict': {
#         'is_sklearn_LR': False
#     }
# }
#
# lr_obj = Invoker.invoke_service('StatsLogisticRegressionExecutor', [encoding_df, y_train], params)
# lr_obj = lr_obj[0]
#
# params = {}
#
# input_list = [encoding_df, lr_obj]
#
# res_list = Invoker.invoke_service('Predictor', input_list, params)
# output_table = res_list[0]
# output_table.to_csv('lr_predict.csv', index=False)
#
# coefficient_matrix = lr_obj.model.params.to_frame()
# coefficient_matrix.columns = ['coe']
# print(coefficient_matrix.reset_index())
#
# encoding_df.reset_index(drop=True, inplace=True)
# import numpy as np
# z_list = []
# for i in range(len(encoding_df.index)):
#     value_list = []
#     for col in encoding_df.columns:
#         value_list.append(coefficient_matrix.loc[col, 'coe'] * encoding_df.loc[i, col])
#     z_list.append(- sum(value_list))
# encoding_df['cal_p'] = 1 / (1 + np.exp(z_list))
#
# from mldesigntoolkit import Invoker
#
# params = {
#         'control_dict': {
#
#         }
#     }
# output_table = output_table.to_frame()
# output_table.reset_index(drop=True, inplace=True)
# output_table.columns = ['pre_p']
# res_list = Invoker.invoke_service('ColumnConcatingHandlerExecutor', [encoding_df, output_table], params)
# p_df = res_list[0]
# p_sorted = p_df.sort_values('pre_p')
# print(p_sorted[['pre_p', 'cal_p']])

# params = {
#     'control_dict': {
#         'score_dict': {
#             'SUM(user_order.amount)_unit_price_binning_15day_ChiMerge6': {
#                 '(72.0, 581.0]': 200,
#                 '(13.0, 72.0]': 200
#             },
#             'MAX(user_order.amount)_unit_price_binning_6month_ChiMerge6': {
#                 '(335.0, 442.0]': 200,
#                 'nan': 200
#             }
#         }
#     }
# }
#
# scorecard_df = Invoker.invoke_service('ScorecardScoreAdjustmentHandlerExecutor', [scorecard_obj], params)
# scorecard_df = scorecard_df[0]
#
# print(scorecard_df.model)
# print(scorecard_df.model)
