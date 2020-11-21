#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/21

from ..modules.modeling import SKlearnLogisticRegression
from ..modules.modeling import StatsLogisticRegression
from ..modules.modeling import SVMClassifier
from ..modules.modeling import RFClassifier
from ..modules.modeling import DTClassifier
from ..modules.modeling import CBDTClassifier
from ..modules.modeling import XgboostClassifier
from ..modules.modeling import LightGBMClassifier
from ..modules.modeling import GBDTClassifier
from ..modules.modeling import OptunaClassifier
from ..modules.modeling import BaggingOptunaClassifier
from ..modules.modeling import ScorecardModel
from ._executor import Executor


class SklearnLogisticRegressionExecutor(Executor):

    def execute(self, input_list):
        model = SKlearnLogisticRegression(**self.control_dict)
        self.outputs.append(model.fit(x_train=input_list[0], y_train=input_list[1]))
        return self.outputs


class StatsLogisticRegressionExecutor(Executor):

    def execute(self, input_list):
        model = StatsLogisticRegression(**self.control_dict)
        self.outputs.append(model.fit(x_train=input_list[0], y_train=input_list[1]))
        return self.outputs


class SVMClassifierExecutor(Executor):

    def execute(self, input_list):
        model = SVMClassifier(**self.control_dict)
        self.outputs.append(model.fit(x_train=input_list[0], y_train=input_list[1]))
        return self.outputs


class RFClassifierExecutor(Executor):

    def execute(self, input_list):
        model = RFClassifier(**self.control_dict)
        self.outputs.append(model.fit(x_train=input_list[0], y_train=input_list[1]))
        return self.outputs


class DTClassifierExecutor(Executor):

    def execute(self, input_list):
        model = DTClassifier(**self.control_dict)
        self.outputs.append(model.fit(x_train=input_list[0], y_train=input_list[1]))
        return self.outputs


class CBDTClassifierExecutor(Executor):

    def execute(self, input_list):
        model = CBDTClassifier(**self.control_dict)
        self.outputs.append(
            model.fit(x_train=input_list[0], y_train=input_list[1], x_valid=input_list[2], y_valid=input_list[3]))
        return self.outputs


class XgboostClassifierExecutor(Executor):

    def execute(self, input_list):
        model = XgboostClassifier(**self.control_dict)
        self.outputs.append(
            model.fit(x_train=input_list[0], y_train=input_list[1], x_valid=input_list[2], y_valid=input_list[3]))
        return self.outputs


class LightGBMClassifierExecutor(Executor):

    def execute(self, input_list):
        model = LightGBMClassifier(**self.control_dict)
        self.outputs.append(
            model.fit(x_train=input_list[0], y_train=input_list[1], x_valid=input_list[2], y_valid=input_list[3]))
        return self.outputs


class GBDTClassifierExecutor(Executor):

    def execute(self, input_list):
        model = GBDTClassifier(**self.control_dict)
        self.outputs.append(
            model.fit(x_train=input_list[0], y_train=input_list[1], x_valid=input_list[2], y_valid=input_list[3]))
        return self.outputs


class OptunaClassifierExecutor(Executor):

    def execute(self, input_list):
        model = OptunaClassifier(**self.control_dict)
        self.outputs.append(model.fit(input_list[0], input_list[1]))
        return self.outputs


class BaggingOptunaClassifierExecutor(Executor):

    def execute(self, input_list):
        model = BaggingOptunaClassifier(**self.control_dict)
        self.outputs.append(model.fit(input_list[0], input_list[1]))
        return self.outputs


class ScorecardModelExecutor(Executor):

    def execute(self, input_list):
        model = ScorecardModel(**self.control_dict)
        self.outputs.append(model.fit(input_list[0], input_list[1]))
        return self.outputs
