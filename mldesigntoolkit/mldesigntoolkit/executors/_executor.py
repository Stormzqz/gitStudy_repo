#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/22
from abc import ABCMeta
from typing import List

from pydantic.main import BaseModel


class Executor(BaseModel, metaclass=ABCMeta):
    control_dict: dict = {}
    outputs: list = []

    @staticmethod
    def execute(input_list):
        pass
