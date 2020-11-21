#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/10/14 16:31
# filename: _using_featuretools
# software: PyCharm

import re
import pandas as pd
import featuretools as ft

from featuretools.variable_types import Timedelta
from pandas import DataFrame
from typing import Dict, List, Optional, Any, Callable

from ..base import BaseEstimator
from ..base import HandlerMixin


class UsingFeaturetoolsHandler(HandlerMixin,
                               BaseEstimator):
    """
        功能： 通过featuretools自动衍生聚合特征
        输入： X_train
        控制：
        输出： 特征衍生后的数据集
    """
    # 数据信息
    entity_id: str = 'id'
    numeric_cols: List[str] = []
    category_cols: Optional[List[str]] = None
    # 基础实体对象参数
    variable_types: Optional[Dict[str, str]] = None
    time_index: Optional[str] = None
    secondary_time_index: Optional[str] = None
    time_already_sorted: bool = False
    # 目标实体对象参数
    additional_variables: Optional[List[str]] = None
    copy_variables: Optional[List[str]] = None
    make_time_index: bool = False
    make_secondary_time_index: Optional[bool] = None
    new_entity_time_index: Optional[str] = None
    new_entity_secondary_time_index: Optional[str] = None
    # 特征衍生循环条件
    window_list: Optional[List[str]] = None
    cutoff_time_dict: Optional[Dict[str, DataFrame]] = None
    # # 特征衍生参数
    agg_primitives: List[str] = ['max', 'min', 'mean', 'sum', 'std', 'count']
    trans_primitives: Optional[List[str]] = None
    groupby_trans_primitives: Optional[List[str]] = None
    allowed_paths: Optional[List[List[str]]] = None
    max_depth: int = 2
    ignore_variables: Optional[Dict[str, List[str]]] = None
    primitive_options: Optional[dict] = None
    drop_contains: Optional[List[str]] = None
    drop_exact: Optional[List[str]] = None
    where_primitives: Optional[List[str]] = None
    max_features: int = -1
    cutoff_time_in_index: bool = False
    approximate: Optional[Timedelta] = None
    chunk_size: Optional[Any] = None  # int or float or None or "cutoff time", optional
    n_jobs: int = 1
    dask_kwargs: Optional[dict] = None
    verbose: bool = False
    return_variable_types: Optional[list] = None  # list[Variable] or str, optional
    progress_callback: Optional[Callable] = None  # callable

    def process(self, X_train):

        if not self.category_cols:
            self.category_cols = [None]
        if not self.window_list:
            self.window_list = [None]
        if not self.cutoff_time_dict:
            self.cutoff_time_dict = {None: None}

        res_matrix = pd.DataFrame()
        for category_column in self.category_cols:
            for cutoff_name, cutoff_time_df in self.cutoff_time_dict.items():
                for window in self.window_list:
                    # 处理没有category_column的情况
                    if category_column:
                        X_train[category_column] = X_train[category_column].astype('category')
                        entity_df = X_train[[self.entity_id, category_column] + self.numeric_cols]
                    else:
                        entity_df = X_train[[self.entity_id] + self.numeric_cols]
                    if self.time_index:
                        entity_df[self.time_index] = X_train[self.time_index]

                    # Featuretools组织实体，实体间关系
                    es = ft.EntitySet(id="entity_set")
                    # 默认子实体没有id列，使用featuretools自动添加自增id
                    es = es.entity_from_dataframe(entity_id='original_dataset',
                                                  dataframe=entity_df,
                                                  index='_id',
                                                  variable_types=self.variable_types,
                                                  make_index=True,
                                                  time_index=self.time_index,
                                                  secondary_time_index=self.secondary_time_index,
                                                  already_sorted=self.time_already_sorted)

                    es = es.normalize_entity(base_entity_id="original_dataset",
                                             new_entity_id=self.entity_id,
                                             index=self.entity_id,
                                             additional_variables=self.additional_variables,
                                             copy_variables=self.copy_variables,
                                             make_time_index=self.make_time_index,
                                             make_secondary_time_index=self.make_secondary_time_index,
                                             new_entity_time_index=self.new_entity_time_index,
                                             new_entity_secondary_time_index=self.new_entity_secondary_time_index)
                    if category_column:
                        es["original_dataset"][category_column].interesting_values = X_train[category_column].unique().tolist()

                    # 执行featuretools特征衍生
                    matrix, _ = ft.dfs(entityset=es,
                                       target_entity=self.entity_id,
                                       cutoff_time=cutoff_time_df,  # 循环项
                                       agg_primitives=self.agg_primitives,
                                       trans_primitives=self.trans_primitives,
                                       groupby_trans_primitives=self.groupby_trans_primitives,
                                       allowed_paths=self.allowed_paths,
                                       max_depth=self.max_depth,
                                       ignore_variables=self.ignore_variables,
                                       primitive_options=self.primitive_options,
                                       drop_contains=self.drop_contains,
                                       drop_exact=self.drop_exact,
                                       where_primitives=self.where_primitives,
                                       max_features=self.max_features,
                                       cutoff_time_in_index=self.cutoff_time_in_index,
                                       training_window=window,  # 循环项
                                       approximate=self.approximate,
                                       chunk_size=self.chunk_size,
                                       n_jobs=self.n_jobs,
                                       dask_kwargs=self.dask_kwargs,
                                       verbose=self.verbose,
                                       return_variable_types=self.return_variable_types,
                                       progress_callback=self.progress_callback
                                       )
                    matrix.columns = [col.replace('original_dataset', '') for col in matrix.columns]
                    matrix.columns = [re.sub(r'[\(\)\[\]\.\s=,]', '_', col) for col in matrix.columns]

                    suffix_list = []
                    if category_column:
                        suffix_list.append(category_column)
                    if cutoff_name:
                        suffix_list.append(cutoff_name)
                    if window:
                        suffix_list.append(window.replace(' ', ''))
                    if suffix_list:
                        matrix.columns = [col + '_' + '_'.join(suffix_list) for col in matrix.columns]

                    matrix.columns = [re.sub(r'_{2,5}', '_', col) for col in matrix.columns]

                    if res_matrix.shape[0] == 0:
                        res_matrix = matrix
                    else:
                        res_matrix = res_matrix.merge(matrix, left_index=True, right_index=True, how='left')

        res_matrix.reset_index(inplace=True)
        return res_matrix
