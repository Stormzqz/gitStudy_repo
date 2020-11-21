#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/9/15
# filename: __init__

from ._bagging_optuna_classifier import BaggingOptunaClassifier
from ._CBDT_classifier import CBDTClassifier
from ._optuna_classifier import OptunaClassifier
from ._GBDT_classifier import GBDTClassifier
from ._lightgbm_classifier import LightGBMClassifier
from ._xgboost_classifier import XgboostClassifier
from ._sklearn_logistic_regression import SKlearnLogisticRegression
from ._stats_logistic_regression import StatsLogisticRegression
from ._DecisionTreeClassifier import DTClassifier
from ._RandomForestClassifier import RFClassifier
from ._SVMClassifier import SVMClassifier
from ._scorecard import ScorecardModel

__init__ = [
    "BaggingOptunaClassifier",
    "OptunaClassifier",
    "GBDTClassifier",
    "LightGBMClassifier",
    "XgboostClassifier",
    "SKlearnLogisticRegression",
    "statsLogisticRegression",
    "DTClassifier",
    "RFClassifier",
    "SVMClassifier",
    "CBDTClassifier",
    "ScorecardModel"
]
