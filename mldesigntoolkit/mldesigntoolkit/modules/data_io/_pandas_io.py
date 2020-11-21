#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/1 11:54
# filename: _pandas_io
# software: PyCharm

import pandas as pd

from typing import List, Optional, Dict
from pydantic import FilePath, PositiveInt

from ..base import BaseEstimator
from ..base import HandlerMixin


class CSVReadingHandler(HandlerMixin,
                        BaseEstimator):
    """
        功能  读取CSV文件
        输入  无
        控制  文件路径        字符串类型  file_path
             日期时间列列表   字符串列表   datetime_columns_list []
             数值列列表      字符串列表   number_columns_list  []
             字符串列列表     字符串列表   string_columns_list []
             是否带有标题行   布尔类型    has_header          True
             编码方式        字符串类型   encoding
             读取行数        整数类型    read_nrows
             读取列名列表     字符串列表   use_columns_list   []
             应用在列上的函数字典  字典    columns_function_dict  {}
        输出
    """

    # 读取文件的路径
    file_path: FilePath
    # 指定的各个特征类型
    datetime_columns_list: List[str] = []
    number_columns_list: List[str] = []
    string_columns_list: List[str] = []
    # 默认数据集带有标题行
    has_header: bool = True
    # 默认编码方式为utf-8
    encoding: str = 'utf-8'
    # 指定读取的行数
    read_nrows: Optional[PositiveInt]
    # 指定读取列名列表
    use_columns_list: Optional[List[str]]
    # 指定应用在列上的函数字典
    columns_function_dict: Optional[Dict[str, str]]

    def process(self):
        header = 0 if self.has_header else None
        df = pd.read_csv(self.file_path, parse_dates=self.datetime_columns_list,
                         header=header, encoding=self.encoding,
                         usecols=self.use_columns_list, nrows=self.read_nrows,
                         converters=self.columns_function_dict)

        # 将指定特征转换为对应的数据格式
        df[self.string_columns_list] = df[self.string_columns_list].astype('str')
        df[self.number_columns_list] = df[self.number_columns_list].astype('float')

        return df
