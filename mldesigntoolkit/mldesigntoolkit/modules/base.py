#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/8/20 16:08
# filename: base
# software: PyCharm

import joblib
import inspect
import warnings

from abc import ABCMeta, abstractmethod
from typing import Any
from pydantic import BaseModel
from collections import defaultdict

from .util import EvaluateGraphic, EvaluateScore


class BaseEstimator(BaseModel, metaclass=ABCMeta):
    """Base class for all estimators in scikit-learn

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always "
                                   "specify their parameters in the signature"
                                   " of their __init__ (no varargs)."
                                   " %s with constructor %s doesn't "
                                   " follow this convention."
                                   % (cls, init_signature))
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn('From version 0.24, get_params will raise an '
                              'AttributeError if a parameter cannot be '
                              'retrieved as an instance attribute. Previously '
                              'it would return None.',
                              FutureWarning)
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self


class HandlerMixin(object):
    """
    一般数据处理基类
    """
    @abstractmethod
    def process(self, df):
        """
        所有子类需实现此数据处理方法
        :param df: 待处理数据集
        :return: 特定数据处理后的数据集
        """
        pass


class RuleMixin(BaseModel, metaclass=ABCMeta):
    """
    数据处理转换规则类型基类
    """
    # 类的实例对象保存的模型对象
    model: Any = None

    @abstractmethod
    def fit(self, X_train):
        """
        所有子类需实现此拟合数据方法
        :param X_train: 待拟合数据集
        :return: 各子类实例对象 self
        """
        pass

    def transform(self, df):
        """
        过滤数据集特征
        :param df: 待过滤特征数据集
        :return: 特征过滤后的数据集
        """
        if not self.model:
            raise RuntimeError("Run 'fit(df)' method firstly please.")
        return df[self.model]


class ModelMixin(BaseModel, metaclass=ABCMeta):
    model: Any = None
    # 模型评估图
    graphic_dict: dict = {}
    # 模型评估分
    score_dict: dict = {}

    # 计算模型评估的数据
    def calculate_score(self, real_y, predict_y):
        # 计算图
        print(self.graphic_dict)
        self.graphic_dict.__setitem__("ROC", EvaluateGraphic.calculate_ROC(real_y, predict_y))
        self.graphic_dict.__setitem__("KSLine", EvaluateGraphic.calculate_ksline(real_y, predict_y))
        self.graphic_dict.__setitem__("PR", EvaluateGraphic.calculate_pr(real_y, predict_y))
        #self.graphic_dict.__setitem__("LIFT", EvaluateGraphic.calculate_lift(real_y, predict_y))
        #self.graphic_dict.__setitem__("GAIN", EvaluateGraphic.calculate_gain(real_y, predict_y))
        # 计算分
        self.score_dict.__setitem__("AUC", EvaluateScore.calculate_AUC(real_y, predict_y))
        self.score_dict.__setitem__("KS", EvaluateScore.calculate_ksscore(real_y, predict_y))
        return self

    def fit(self, **kwargs):
        pass

    def predict(self, x):
        return self.model.predict_proba(x)[:, 1]

    def save(self, path):
        joblib.dump(self, str(path))

    @staticmethod
    def load(path):
        return joblib.load(path)
