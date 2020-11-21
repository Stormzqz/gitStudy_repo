#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/18
import math

from sklearn.metrics import roc_curve, precision_recall_curve
import pandas as pd


class EvaluateGraphic:
    @staticmethod
    def calculate_ROC(y_test, y_pred):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        data = {'False Positive Rate': fpr,
                'True Positive Rate': tpr}
        roc = pd.DataFrame(data=data, columns=('False Positive Rate', 'True Positive Rate'))
        return roc

    @staticmethod
    def calculate_ksline(y_test, y_pred):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        data = {'False Positive Rate': fpr,
                'True Positive Rate': tpr,
                'thresholds': thresholds}
        ksline = pd.DataFrame(data=data,
                              columns=('False Positive Rate', 'True Positive Rate', 'thresholds'))
        ksline.loc[:,'KS_VALUE'] = ksline.loc[:,'False Positive Rate'] - ksline.loc[:,'True Positive Rate']
        ksline.loc[:, 'KS_VALUE'] = ksline.loc[:,'KS_VALUE'].apply(lambda x:abs(x))
        return ksline

    @staticmethod
    def calculate_pr(y_test, y_pred):
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        data = {'precision': precision,
                'recall': recall}
        pr = pd.DataFrame(data=data, columns=('precision', 'recall'))
        return pr

    @staticmethod
    def calculate_lift(y_test, y_pred):
        set_list = y_pred.values.tolist()
        set_list.sort(reverse=True)
        # 数据量过大（大于2000）时，将数据压缩为1000
        set_list_new = []
        if len(set_list) >= 2000:
            interval = math.floor(len(set_list) / 1000)
            for k in range(math.floor(len(set_list) / interval)):
                if k * interval <= len(set_list):
                    set_list_new.append(set_list[k * interval])
            set_list_new.append(set_list[-1])
        else:
            # 小于2000时，使用源数据
            set_list_new = set_list
        set_list_new.sort(reverse=True)
        # 根据混淆矩阵，计算各项指标
        # threshList = []
        # TPRList = []    #灵敏度，查全率，也叫召回率(正例分对的概率)
        precision_list = []  # 精确率,也叫查准率
        # FPRList = []     #1-特异度（负例分错的概率）
        lift_list = []
        predict_1_ratio_list = []
        # 以各分数（分箱分数）为阈值，准确率（Gain）/精度/Lift曲线值
        for thresh in set_list_new:
            TP = sum((y_pred.loc[:, 0] >= thresh) & (y_test.loc[:, 0] == 1))
            FP = sum((y_pred.loc[:, 0] >= thresh) & (y_test.loc[:, 0] == 0))
            TN = sum((y_pred.loc[:, 0] < thresh) & (y_test.loc[:, 0] == 0))
            FN = sum((y_pred.loc[:, 0] < thresh) & (y_test.loc[:, 0] == 1))
            # threshList.append(thresh)
            # TPRList.append(TP * 1.0 / (TP + FN))
            precision_list.append(TP * 1.0 / (TP + FP))
            # FPRList.append(FP * 1.0 / (FP + TN))
            predict_1_ratio_list.append((TP + FP) * 1.0 / (TP + FP + FN + TN))
            lift_list.append((TP * 1.0 / (TP + FP)) / ((TP + FN) * 1.0 / (TP + FP + FN + TN)))
        # 封装整理返回值
        # lift_df = pd.DataFrame({'Depth': predict_1_ratio_list, 'Lift': lift_list, 'threshList': threshList})
        lift_df = pd.DataFrame({'Depth': predict_1_ratio_list, 'Lift': lift_list})
        lift_df.sort_values(by='Depth', ascending=True, inplace=True)
        lift_df = lift_df.reset_index(drop=True)

        return lift_df

    @staticmethod
    def calculate_gain(y_test, y_pred):
        set_list = y_pred.values.tolist()
        set_list.sort(reverse=True)
        # 数据量过大（大于2000）时，将数据压缩为1000
        set_list_new = []
        if len(set_list) >= 2000:
            interval = math.floor(len(set_list) / 1000)
            for k in range(math.floor(len(set_list) / interval)):
                if k * interval <= len(set_list):
                    set_list_new.append(set_list[k * interval])
            set_list_new.append(set_list[-1])
        else:
            # 小于2000时，使用源数据
            set_list_new = set_list
        set_list_new.sort(reverse=True)
        # 根据混淆矩阵，计算各项指标
        # threshList = []
        # TPRList = []    #灵敏度，查全率，也叫召回率(正例分对的概率)
        precision_list = []  # 精确率,也叫查准率
        # FPRList = []     #1-特异度（负例分错的概率）
        lift_list = []
        predict_1_ratio_list = []
        # 以各分数（分箱分数）为阈值，准确率（Gain）/精度/Lift曲线值
        for thresh in set_list_new:
            TP = sum((y_pred.loc[:, 0] >= thresh) & (y_test.loc[:, 0] == 1))
            FP = sum((y_pred.loc[:, 0] >= thresh) & (y_test.loc[:, 0] == 0))
            TN = sum((y_pred.loc[:, 0] < thresh) & (y_test.loc[:, 0] == 0))
            FN = sum((y_pred.loc[:, 0] < thresh) & (y_test.loc[:, 0] == 1))
            # threshList.append(thresh)
            # TPRList.append(TP * 1.0 / (TP + FN))
            precision_list.append(TP * 1.0 / (TP + FP))
            # FPRList.append(FP * 1.0 / (FP + TN))
            predict_1_ratio_list.append((TP + FP) * 1.0 / (TP + FP + FN + TN))
            lift_list.append((TP * 1.0 / (TP + FP)) / ((TP + FN) * 1.0 / (TP + FP + FN + TN)))
        # Gain图(精准度precision图)
        gain_df = pd.DataFrame({'Depth': predict_1_ratio_list, 'Gain': precision_list})
        gain_df.sort_values(by='Depth', ascending=True, inplace=True)
        gain_df = gain_df.reset_index(drop=True)

        return gain_df
