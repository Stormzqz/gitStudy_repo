#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/18
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np


class EvaluateScore:
    @staticmethod
    def calculate_AUC(y_test, y_pred):
        auc = roc_auc_score(y_test, y_pred)
        if auc < 0.5:
            auc = 1 - auc
        return auc

    @staticmethod
    def calculate_ksscore(y_test, y_pred):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        ksscore = np.max(abs(fpr - tpr))
        return ksscore

    def calculate_gini(self, y_test, y_pred):
        pass
