#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Mark
# datetime: 2020/10/14 14:24
# filename: _scoredcard_score_adjustment
# software: PyCharm

from typing import Dict

from ..base import BaseEstimator
from ..base import HandlerMixin


class ScorecardScoreAdjustmentHandler(HandlerMixin,
                                      BaseEstimator):
    """
        功能： 修改评分卡内具体特征分箱对应的分数，支持批量修改
        输入： 评分卡模型 scorecard_df
        控制： 被修改分数字典：{column_name: {binning: score}} score_dict
        输出： 修改分数后的评分卡数据表
    """

    # 被修改分数字典：{column_name: {binning: score}}
    score_dict: Dict[str, Dict[str, float]] = []

    def process(self, scorecard_model):
        scorecard_df = scorecard_model.model
        scorecard_df['binning'] = scorecard_df['binning'].astype('str')
        for column, value in self.score_dict.items():
            for binning, score in value.items():
                scorecard_df.loc[(scorecard_df.loc[:, 'column_name'] == column) & (
                            scorecard_df.loc[:, 'binning'] == binning), 'score'] = score
        scorecard_model.model = scorecard_df
        return scorecard_model
