#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/23

from ._executor import Executor
from ..modules.data_io import CSVReadingHandler
from ..modules.data_manipulation import RowPartitioningHandler
from ..modules.data_manipulation import ColumnConcatingHandler
from ..modules.data_manipulation import ColumnTypeConvertingHandler
from ..modules.data_manipulation import ColumnTypeFilteringHandler
from ..modules.data_manipulation import ColumnRemoveHandler
from ..modules.data_manipulation import DropAffixHandler
from ..modules.data_manipulation import ColumnNameFilteringHandler
from ..modules.data_manipulation import DataframesFeatureFilteringHandlar
from ..modules.data_manipulation import ToDataframeHandler
from ..modules.data_manipulation import RowCleaningHandler
from ..modules.data_manipulation import RowConcatingHandler
from ..modules.data_manipulation import MonotonicitySelectionRuleDataHandler
from ..modules.data_manipulation import ScorecardModelUpdateHandler
from ..modules.data_manipulation import ScorecardScoreAdjustmentHandler
from ..modules.data_manipulation import UsingFeaturetoolsHandler
from ..modules.data_manipulation import GeneratingStdColumnsHandler
from ..modules.feature_engineering import TwoDatetimeColumnsSubtractHandler
from ..modules.feature_engineering import DatetimeColumnDecomposeHandler
from ..modules.feature_engineering import CategoryColumnsComposeHandler
from ..modules.feature_engineering import NumberColumnsCalculatingHandler


class CSVReadingHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = CSVReadingHandler(**self.control_dict)
        self.outputs.append(handler.process())
        return self.outputs


class ColumnConcatingHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = ColumnConcatingHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0], input_list[1]))
        return self.outputs


class ColumnTypeConvertingHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = ColumnTypeConvertingHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class ColumnTypeFilteringHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = ColumnTypeFilteringHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class ColumnNameFilteringHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = ColumnNameFilteringHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class ColumnRemoveHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = ColumnRemoveHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class DropAffixHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = DropAffixHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class DataframesFeatureFilteringHandlarExecutor(Executor):

    def execute(self, input_list):
        handler = DataframesFeatureFilteringHandlar(**self.control_dict)
        self.outputs.append(handler.process(input_list[0], input_list[1]))
        return self.outputs


class ToDataframeHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = ToDataframeHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class RowCleaningHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = RowCleaningHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class TwoDatetimeColumnsSubtractHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = TwoDatetimeColumnsSubtractHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class DatetimeColumnDecomposeHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = DatetimeColumnDecomposeHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class NumberColumnsCalculatingHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = NumberColumnsCalculatingHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class CategoryColumnsComposeHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = CategoryColumnsComposeHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class RowPartitioningHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = RowPartitioningHandler(**self.control_dict)
        self.outputs = list(handler.process(input_list[0]))
        return self.outputs


class RowConcatingHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = RowConcatingHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0], input_list[1]))
        return self.outputs


class MonotonicitySelectionRuleDataHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = MonotonicitySelectionRuleDataHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0], input_list[1]))
        return self.outputs


class ScorecardModelUpdateHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = ScorecardModelUpdateHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0], input_list[1]))
        return self.outputs


class ScorecardScoreAdjustmentHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = ScorecardScoreAdjustmentHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class UsingFeaturetoolsHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = UsingFeaturetoolsHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs


class GeneratingStdColumnsHandlerExecutor(Executor):

    def execute(self, input_list):
        handler = GeneratingStdColumnsHandler(**self.control_dict)
        self.outputs.append(handler.process(input_list[0]))
        return self.outputs
