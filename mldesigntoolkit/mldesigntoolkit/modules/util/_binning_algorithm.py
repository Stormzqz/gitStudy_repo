#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/9/15 9:57
# filename: _binning_algorithm
# software: PyCharm

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import chisquare, chi2_contingency


class ChiMerge(object):

    def __init__(self, column=None, bin_size_threshold=5, confidence_val=3.841, num_of_bins=10):
        self.column = column
        self.bin_size_threshold = bin_size_threshold
        self.confidence_val = confidence_val
        self.num_of_bins = num_of_bins
        self._dim = None
        self.bins = None

    def fit(self, df, label):
        """
        卡方分箱拟合数据
        :param df: 带有标签的待拟合数据集
        :param label: 标签列名
        :return: 当前类的实例变量
        """
        self._dim = df.shape[1]
        self.bins = self._do_chimerge(df=df, label=label)
        return self

    def transform(self, df):
        """
        执行数据分箱
        :param df: 待分箱数据集
        :return: 分箱后的数据集
        """
        if self.bins is not None:
            df[self.column + '_chimerge'] = pd.cut(df[self.column], bins=self.bins, include_lowest=False)

        return df

    def _calculate_chimerge(self, bin_df, row_idx):
        """
        使用化简后的公式计算卡方值（已验证）
        :param bin_df: 分箱数据集
        :param row_idx: 分箱行索引
        :return: 索引行分箱与下一行分箱的卡方值
        """
        # 保留可选调用，效率偏低
        # d1 = bin_df[[row_idx], [1, 2]].flatten()
        # d2 = bin_df[[row_idx + 1], [1, 2]].flatten()
        # data = np.array([d1, d2])
        # data_exp = chi2_contingency(data)
        # data_list = np.array(data).reshape(4, 1)
        # exp_list = data_exp[3].reshape(4, 1)
        # chi2 = chisquare(data_list, f_exp=exp_list).statistic
        # return chi2

        # 除数
        divisor = (((bin_df[row_idx, 1] * bin_df[row_idx + 1, 2] - bin_df[row_idx, 2] * bin_df[row_idx + 1, 1]) ** 2) *
                   (bin_df[row_idx, 1] + bin_df[row_idx, 2] + bin_df[row_idx + 1, 1] + bin_df[row_idx + 1, 2]))
        # 被除数
        divedend = ((bin_df[row_idx, 1] + bin_df[row_idx, 2]) * (bin_df[row_idx + 1, 1] + bin_df[row_idx + 1, 2]) *
                    (bin_df[row_idx, 1] + bin_df[row_idx + 1, 1]) * (bin_df[row_idx, 2] + bin_df[row_idx + 1, 2]))
        if divedend == 0:
            # 如果被除数为0，返回0作为卡方值。执行合并分箱
            chi = 0.0
        else:
            # 计算卡方值
            chi = divisor / divedend
        return chi

    def _combine_bins(self, bins_np, min_index):
        """
        合并分箱
        :param bins_np: 分箱表
        :param min_index: 待合并行索引
        :return: 合并后的分箱表
        """
        # 最后一个分箱向上合并
        if min_index + 1 == len(bins_np):
            min_index = min_index - 1
        # 执行分箱合并
        bins_np[min_index, 1] = bins_np[min_index, 1] + bins_np[min_index + 1, 1]
        bins_np[min_index, 2] = bins_np[min_index, 2] + bins_np[min_index + 1, 2]
        bins_np[min_index, 3] = bins_np[min_index, 3] + bins_np[min_index + 1, 3]
        bins_np[min_index, 0] = bins_np[min_index + 1, 0]
        # 删除被合并的分箱
        bins_np = np.delete(bins_np, min_index + 1, 0)
        return bins_np

    def _do_chimerge(self, df, label):
        """
        计算卡方分箱
        参考：https://cloud.tencent.com/developer/article/1418720
              https://zhuanlan.zhihu.com/p/115267395
        :param df: 待计算卡方分箱的数据集，包含标签列
        :param label: 标签列名称
        :return: 分箱数据
        """
        # 构建初始分箱df：bins_np
        label_str = label + '_str'
        df[label_str] = df[label].apply(lambda x: 'neg' if x else 'pos')
        bins_df = pd.pivot_table(df, index=self.column, columns=[label_str], values=[label], aggfunc=[np.size],
                                 fill_value=0).reset_index()
        bins_df.columns = [self.column, 'neg', 'pos']

        bins_np = np.array(bins_df)
        # 初始合并分箱操作，避免计算期望时除零
        i = 0
        while i <= bins_np.shape[0] - 2:
            if (bins_np[i, 1] == 0 and bins_np[i + 1, 1] == 0) or (bins_np[i, 2] == 0 and bins_np[i + 1, 2] == 0):
                bins_np[i, 1] = bins_np[i, 1] + bins_np[i + 1, 1]  # pos
                bins_np[i, 2] = bins_np[i, 2] + bins_np[i + 1, 2]  # neg
                bins_np[i, 0] = bins_np[i + 1, 0]
                bins_np = np.delete(bins_np, i + 1, 0)
                i = i - 1
            i = i + 1

        # 添加各分箱数据计数列
        records_sum = bins_np[:, 1] + bins_np[:, 2]
        records_sum = records_sum.reshape(len(records_sum), 1)
        bins_np = np.append(bins_np, records_sum, axis=1)

        # 计算卡方值：
        # ∑[(yA-yB)²/yB]
        chimerge_np = np.array([])
        for row_index in np.arange(bins_np.shape[0] - 1):
            chi = self._calculate_chimerge(bins_np, row_index)
            chimerge_np = np.append(chimerge_np, chi)

        # 执行卡方分箱
        while True:
            # 停止条件: 小于给定分箱数并且最小卡方值大于阈值
            # 卡方阈值的确定：
            # 　　根据显著性水平和自由度得到卡方值自由度比类别数量小1。
            #     例如：有3类, 自由度为2，则90 % 置信度(10 % 显著性水平)下，卡方的值为4.6。
            if len(chimerge_np) <= (self.num_of_bins - 1):  # and min(chimerge_np) >= self.confidence_val:
                break

            min_chi_index = np.argwhere(chimerge_np == min(chimerge_np))[0]
            # 合并卡方值最小的分箱
            bins_np = self._combine_bins(bins_np, min_chi_index)

            # 更新相邻分箱卡方值
            if min_chi_index == bins_np.shape[0] - 1:
                # 合并了最后一个分箱，更新上一个卡方值
                chimerge_np[min_chi_index - 1] = self._calculate_chimerge(bins_np, min_chi_index - 1)
                chimerge_np = np.delete(chimerge_np, min_chi_index, axis=0)
            elif min_chi_index == 0:
                # 合并了第二个分箱，更新当前卡方值
                chimerge_np[min_chi_index] = self._calculate_chimerge(bins_np, min_chi_index)
                chimerge_np = np.delete(chimerge_np, min_chi_index + 1, axis=0)
            else:
                # 合并了中间的分箱，相邻的卡方值均要跟新
                chimerge_np[min_chi_index - 1] = self._calculate_chimerge(bins_np, min_chi_index - 1)
                chimerge_np[min_chi_index] = self._calculate_chimerge(bins_np, min_chi_index)
                chimerge_np = np.delete(chimerge_np, min_chi_index + 1, axis=0)

        # 处理分箱内数据量过小问题
        while True:
            min_value = min(bins_np[:, 3])
            if min_value >= self.bin_size_threshold:
                break
            min_index = np.argwhere(bins_np[:, 3] == min_value).flatten()[0]
            # 合并分箱内数据量小于阈值的分箱
            bins_np = self._combine_bins(bins_np, min_index)

        # 控制台显示分箱结果
        show_bins_df = pd.DataFrame()
        show_bins_df['variable'] = [self.column] * bins_np.shape[0]
        show_tmp = []
        # 整理分箱对象
        bins = [df[self.column].min() - 0.1]
        for row_index in np.arange(bins_np.shape[0]):
            bins.append(bins_np[row_index, 0])

            if row_index == 0:
                bin = '-inf' + ',' + str(bins_np[row_index, 0])
            elif row_index == bins_np.shape[0] - 1:
                bin = str(bins_np[row_index - 1, 0]) + ', inf'
            else:
                bin = str(bins_np[row_index - 1, 0]) + ',' + str(bins_np[row_index, 0])
            show_tmp.append(bin)

        # 处理分箱内数据量过小问题
        # while True:
        #     test_res = pd.cut(df[self.column], bins=bins, include_lowest=False).value_counts()
        #     test_res = test_res.reset_index()
        #     test_res.sort_values(by=self.column, inplace=True)
        #     if test_res.iloc[0, 1] < self.bin_size_threshold:
        #         # 删除数据量过小的分箱
        #         value = test_res.iloc[0, 0].right
        #         bins.remove(value)
        #     else:
        #         break

        # 控制台输出分箱结果
        show_bins_df['interval'] = show_tmp
        show_bins_df['flag_0'] = bins_np[:, 2]
        show_bins_df['flag_1'] = bins_np[:, 1]
        print()
        print('Interval for variable %s' % self.column)
        print(show_bins_df)
        print(bins)

        return bins


class DiscretizeByDecisionTree(object):

    def __init__(self, col=None, min_samples_leaf=0.05, max_depth=10, max_leaf_nodes=6, tree_model='gini'):
        self.col = col
        self.min_samples_leaf = min_samples_leaf
        self._dim = None
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.tree_model = tree_model
        self.bins = None

    def __optimal_binning_boundary(self, x: pd.Series, y: pd.Series, nan: float = -999.) -> list:
        """
        利用决策树获得最优分箱的边界值列表
        :param x:
        :param y:
        :param nan:
        :return:
        """
        boundary = []  # 待return的分箱边界值列表

        x = x.fillna(nan).values  # 填充缺失值

        clf = DecisionTreeClassifier(criterion=self.tree_model,  # “基尼系数”最小化准则划分
                                     max_leaf_nodes=self.max_leaf_nodes,  # 最大叶子节点数
                                     max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf)  # 叶子节点样本数量最小占比

        clf.fit(x.reshape(-1, 1), y)  # 训练决策树

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
                boundary.append(threshold[i])

        boundary.sort()

        min_x = x.min()
        max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
        boundary = [min_x] + boundary + [max_x]

        return boundary

    def fit(self, x_train: pd.DataFrame, y_train):
        nan: float = -999.

        boundary = self.__optimal_binning_boundary(x_train[self.col], y_train, nan)  # 获得最优分箱边界值列表

        self.bins = boundary

        return self

    def transform(self, df):
        if self.bins is not None:
            df[self.col + '_tree'] = pd.cut(df[self.col], bins=self.bins, include_lowest=False)
        return df
