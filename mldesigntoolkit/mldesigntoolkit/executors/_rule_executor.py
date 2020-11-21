#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/23

from ..modules.data_manipulation import ColumnCleaningRule, MissingValuesFillingRule, OutlierHandlingRule, \
    HighAndLowCardinalitySplittingRule
from ..modules.feature_engineering import ColumnBinningRule, SupervisedColumnBinningRule, ColumnEncodingRule, \
    SupervisedColumnEncodingRule, FeatureSelectionByTreeAlgorithmRule, FeatureSelectionByCorrelationRule, \
    FeatureSelectionByMonotonicityRule, FeatureSelectionByVIFRule, FeatureSelectionByIVRule, \
    FeatureSelectionByPValueAuc, FeatureSelectionandBinning
from ._executor import Executor


class ColumnCleaningRuleExecutor(Executor):

    def execute(self, input_list):
        rule = ColumnCleaningRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0]))
        return self.outputs


class MissingValuesFillingRuleExecutor(Executor):

    def execute(self, input_list):
        rule = MissingValuesFillingRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0]))
        return self.outputs


class OutlierHandlingRuleExecutor(Executor):

    def execute(self, input_list):
        rule = OutlierHandlingRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0]))
        return self.outputs


class ColumnBinningRuleExecutor(Executor):

    def execute(self, input_list):
        rule = ColumnBinningRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0]))
        return self.outputs


class SupervisedColumnBinningRuleExecutor(Executor):

    def execute(self, input_list):
        rule = SupervisedColumnBinningRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0], input_list[1]))
        return self.outputs


class ColumnEncodingRuleExecutor(Executor):

    def execute(self, input_list):
        rule = ColumnEncodingRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0]))
        return self.outputs


class SupervisedColumnEncodingRuleExecutor(Executor):

    def execute(self, input_list):
        rule = SupervisedColumnEncodingRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0], input_list[1]))
        return self.outputs


class FeatureSelectionByTreeAlgorithmRuleExecutor(Executor):

    def execute(self, input_list):
        rule = FeatureSelectionByTreeAlgorithmRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0], input_list[1]))
        return self.outputs


class FeatureSelectionByCorrelationRuleExecutor(Executor):

    def execute(self, input_list):
        rule = FeatureSelectionByCorrelationRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0], input_list[1]))
        return self.outputs


class FeatureSelectionByMonotonicityRuleExecutor(Executor):

    def execute(self, input_list):
        rule = FeatureSelectionByMonotonicityRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0]))
        return self.outputs


class FeatureSelectionByIVRuleExecutor(Executor):

    def execute(self, input_list):
        rule = FeatureSelectionByIVRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0], input_list[1]))
        return self.outputs


class FeatureSelectionByVIFRuleExecutor(Executor):

    def execute(self, input_list):
        rule = FeatureSelectionByVIFRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0]))
        return self.outputs


class HighAndLowCardinalitySplittingRuleExecutor(Executor):

    def execute(self, input_list):
        rule = HighAndLowCardinalitySplittingRule(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0]))
        return self.outputs


class FeatureSelectionByPValueAucExecutor(Executor):

    def execute(self, input_list):
        rule = FeatureSelectionByPValueAuc(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0], input_list[1]))
        return self.outputs


class FeatureSelectionandBinningExecutor(Executor):

    def execute(self, input_list):
        rule = FeatureSelectionandBinning(**self.control_dict)
        self.outputs.append(rule.fit(input_list[0], input_list[1]))
        return self.outputs
