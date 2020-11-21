#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: nanata
# datetime: 2020/09/22

from ._executor import Executor


class Transformer(Executor):

    def execute(self, input_list):
        model_obj = input_list[1]
        for attr, value in self.control_dict.items():
            if hasattr(model_obj, attr):
                setattr(model_obj, attr, value)
        self.outputs.append(model_obj.transform(input_list[0]))
        return self.outputs
