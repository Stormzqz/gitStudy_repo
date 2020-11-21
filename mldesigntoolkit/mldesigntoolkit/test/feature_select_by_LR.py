#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/10/22 14:33
# filename: feature_select_by_LR
# software: PyCharm

import os
import time
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from mldesigntoolkit.modules.modeling import StatsLogisticRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 130)

main_test_auc_dict = {}


def get_col_list_by_p_value(filtered_col_list, X_train, y_train, p_value_threshold=0.05):
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
            # coefficient_matrix = statslogistic.model.params
            p_values = statslogistic.model.pvalues
            # P值筛选截止条件：所有特征的 P 值均小于给定阈值
            if step_2 == 0 and p_values.apply(lambda x: x <= p_value_threshold).all():
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

    return filtered_col_list


def loop_columns(column_list, file, X_train, X_test, y_train, y_test):
    # 递归停止条件
    if len(column_list) <= 4:
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
        filtered_col_list = get_col_list_by_p_value(cols, X_train, y_train)
        filtered_col_set = set(filtered_col_list)
        if filtered_col_set in finded_column_list:
            # 如果是已有组合，就pass
            pass
        else:
            finded_column_list.append(filtered_col_set)
            # 计算auc
            statslogistic = StatsLogisticRegression()
            statslogistic.fit(x_train=[filtered_col_list], y_train=y_train)
            y_train_predict = statslogistic.predict(X_train[filtered_col_list])
            y_test_predict = statslogistic.predict(X_test[filtered_col_list])
            train_auc = roc_auc_score(y_train, y_train_predict)
            test_auc = roc_auc_score(y_test, y_test_predict)
            # 记录 结果
            col_dict = {'train_auc': train_auc, 'test_auc': test_auc,
                        'filtered_columns': filtered_col_list, 'input_columns': column_list}
            # 写文件
            file.write('\n')
            file.write(json.dumps(col_dict))
            print(json.dumps(col_dict))
            # 保存结果，后续筛选auc最大值
            test_auc_dict[test_auc] = col_dict
    # 汇集auc，用于最后的判断
    main_test_auc_dict.update(test_auc_dict)
    # 递归，列数一定要减少
    test_max_auc = max(test_auc_dict.keys())
    test_column_list = test_auc_dict[test_max_auc]['filtered_columns']
    if len(test_column_list) == len(column_list):
        test_auc_dict.pop(test_max_auc)
        test_max_auc = max(test_auc_dict.keys())
        test_column_list = test_auc_dict[test_max_auc]['filtered_columns']

    loop_columns(test_column_list, file, X_train, X_test, y_train, y_test)


def main():
    bath_path = os.getcwd()
    df = pd.read_csv(os.path.join(bath_path, r'test_data\LR_data.csv'))[
        ['label', 'SOURCE_TYPE_woe', 'HOUSE_PROPERTY_woe', 'RETAIL_TYPE_NEW_woe', 'CUST_SIZE_NEW_woe',
         'age_ChiMerge5_woe', 'IDENTITY_MONTHS_ChiMerge4_woe', 'YITU_SIMILARITY_ChiMerge6_woe',
         'TOBACCO_END_MONTH_ChiMerge6_woe', 'register_capital_new_ChiMerge6_woe', 'AMOUNT_7_12_std_ChiMerge6_woe',
         'AMOUNT_12_std_ChiMerge6_woe', 'TOTAL_PRICE_3_std_ChiMerge6_woe', 'TOTAL_PRICE_4_6_std_ChiMerge6_woe',
         'TOTAL_PRICE_6_std_ChiMerge5_woe', 'TOTAL_PRICE_7_12_std_ChiMerge6_woe', 'TOTAL_PRICE_12_std_ChiMerge6_woe',
         'UNIT_PRICE_new_6_std_ChiMerge4_woe',
         'SUM_AMOUNT_current_3month_subtract_SUM_AMOUNT_up4month_3month_ChiMerge6_woe',
         'SUM_AMOUNT_current_6month_subtract_SUM_AMOUNT_up7month_6month_ChiMerge6_woe']]
    y = df.pop('label')
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    column_list = X_train.columns.tolist()

    with open('result.txt', 'a') as file:
        file.write('\n')
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        loop_columns(column_list, file, X_train, X_test, y_train, y_test)

        test_max_auc = max(main_test_auc_dict.keys())
        file.write('\nmax auc:\n')
        file.write(json.dumps(main_test_auc_dict[test_max_auc]))
        print(json.dumps(main_test_auc_dict[test_max_auc]))

        # 转换为dataframe
        df_dict = {}
        train_auc_list = []
        test_auc_list = []
        filtered_columns_list = []
        input_columns_list = []
        for _, values in main_test_auc_dict.items():
            train_auc_list.append(values['train_auc'])
            test_auc_list.append(values['test_auc'])
            filtered_columns_list.append(str(values['filtered_columns']))
            input_columns_list.append(str(values['input_columns']))
        df_dict['train_auc'] = train_auc_list
        df_dict['test_auc'] = test_auc_list
        df_dict['filtered_columns'] = filtered_columns_list
        df_dict['input_columns'] = input_columns_list
        res_df = pd.DataFrame(df_dict)
        return res_df


if __name__ == "__main__":
    res_dict = main()
    print(res_dict.head(n=None))
