#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: mark
# datetime: 2020/1/14 11:20
# software: PyCharm

import importlib
import os


class Singleton(type):
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class Invoker(metaclass=Singleton):
    list_len_error = "The given input_df_list length must be {}"
    service_error = "The given service: {} is not in service list"
    base_dir = os.path.dirname(__file__)

    def call_service_process(self, service_name, input_dfs, input_control_dict):
        # invoke service using python reflection.
        modules = importlib.import_module('.'.join(['mldesigntoolkit', 'executors']))
        if hasattr(modules, service_name):
            service = getattr(modules, service_name)
            # service object instantiation.
            service_obj = service(**input_control_dict)
            # run the service.
            res = service_obj.execute(input_dfs)
        else:
            raise ValueError(self.service_error.format(service_name))
        return res

    @staticmethod
    def invoke_service(service_name, input_df_list, conf_dict):
        # service = ServiceInvoking.get_instance()
        invoker = Invoker()

        # 切面点

        output_list = invoker.call_service_process(service_name, input_df_list, conf_dict)

        # 切面点

        return output_list


if __name__ == "__main__":
    # 调用示例
    from mldesigntoolkit import Invoker

    params = {
        'control_dict': {
            'file_path': r'E:\IdeaProjects\mldesigntoolkit\mldesigntoolkit\test\test_data\adultData_drop_id.csv'
        }
    }

    outputs = Invoker.invoke_service('CSVReadingHandlerExecutor', [], params)
    output_table = outputs[0]
    print(output_table)
