#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/21

from ._executor import Executor


class Predictor(Executor):

    def execute(self, input_list):
        self.outputs.append(input_list[1].predict(input_list[0]))
        return self.outputs
