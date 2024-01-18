#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np

class Metrics(object):
    def __init__(self):
        super(Metrics, self).__init__()

    # for each task
    # traj_level_acc: [rank0, rank1, rank2, ..., rankn]: the predicted rank of ground truth for each traj

    # larger acc value, the more predicted trajs in top-k, the better prediction performance
    def acc_score(self, traj_level_acc, k=5):
        top_k_count = traj_level_acc.count(rk for rk in traj_level_acc if rk <= k)
        return top_k_count / len(traj_level_acc)  # number of trajs in this task

    # larger mrr value, the more smaller rank values, the better prediction performance
    def mrr_score(self, traj_level_acc):
        traj_num = len(traj_level_acc)
        recip_traj_level_acc = [1 / rank for rank in traj_level_acc]
        mrr = 1 / traj_num * sum(recip_traj_level_acc)
        return mrr

    # task_score_weights: [traj_num0, traj_num1, traj_num2, ..., traj_numn]: the number of trajs in each task
    # Return: task_scores
    # Acc@N = hit@N / T
    # hit@N: the number of testing trajectories with ground truths at the top-N prediction results
    # T: the number of total testing trajectories
    # MRR = 1/T * sum(1/rank_i with i from 1 to |T|)
    # rank_i: the predicted rank of the ground truth in the predicted result for testing trajectory i
    def compute_scores(self, traj_level_acc):
        scores_results = []
        k_values = [20, 40, 60]

        for k_i in k_values:
            acc_k_i = self.acc_score(traj_level_acc, k_i)
            scores_results.append(acc_k_i)

        mrr = self.mrr_score(traj_level_acc)
        scores_results.append(mrr)
        return scores_results

















