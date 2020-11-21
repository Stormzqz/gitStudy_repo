#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/21

from ._executor import Executor


class Evaluator(Executor):

    def execute(self, input_list):
        self.outputs.append(input_list[2].calculate_score(input_list[0], input_list[1]))
        return self.outputs
