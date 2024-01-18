#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Step 4: get same origin and destination trajectories for each trajectory (for each user)

Considering the complexity of:
1. find trajectories with same o and d for each target trajectory
2. find corresponding times file and lengths file
3. pad the three types of data

Thus, we change to get same origin trajectories for each trajectory (for each user)
'''

import pickle
import time
import json
import numpy as np
from constants import *
import os


def read_sameo_data():
    with open(hisdataset_path + 'sameo_user.pkl', 'rb') as f1:
        sameo_user = pickle.load(f1)
    with open(hisdataset_path + 'sameo_trajs.json', 'r') as f2:
        sameo_trajs = json.load(f2)
    with open(hisdataset_path + 'sameo_time.json', 'r') as f3:
        sameo_times = json.load(f3)
    with open(hisdataset_path + 'sameo_length.json', 'r') as f4:
        sameo_lengths = json.load(f4)
    return sameo_user, sameo_trajs, sameo_times, sameo_lengths


# Input: a target trajectory in mtrain/valid/test_tasks
# target traj A, user id U -> origin cell id o and destination cell id d -> sameo trajs B, user ids Us
# judge whether U exists in Us, if exists, extract corresponding trajs in B and filter trajs with destination cell id
def get_sameo_data(target_user_id, target_traj, sameo_user, sameo_trajs, sameo_times, sameo_lengths):
    target_o = target_traj[0]
    # target_d = target_traj[-1]

    same_o_trajs = []
    same_o_times = []
    same_o_lengths = []

    targeto_users = sameo_user[target_o]
    targeto_trajs = sameo_trajs[target_o]
    targeto_times = sameo_times[target_o]
    targeto_lengths = sameo_lengths[target_o]

    sameuser_indices = [i for i, user in enumerate(targeto_users) if user == target_user_id]
    # target user exists in sameo_user[target_o]
    if sameuser_indices:
        for idx in sameuser_indices:
            sameuser_traj = targeto_trajs[idx]
            # if sameuser_traj[-1] == target_d:
            same_o_trajs.append(sameuser_traj)
            same_o_times.append(targeto_times[idx])
            same_o_lengths.append(targeto_lengths[idx])

    return same_o_trajs, same_o_times, same_o_lengths
    # not exist
    # else:
    #     return None


# Return: user id -> [0] spt info, [1] qry info
# [0]/[1]: traj -> same_o_trajs, same_o_times, same_o_lengths
def get_task_data(mtype_tasks, sameo_user, sameo_trajs, sameo_times, sameo_lengths):
    # dictionary format: user id -> [0] spt, [1] qry
    sameo_tasks_data = {}
    # user_ids = mtype_tasks.keys()

    for user_id, all_info in mtype_tasks.items():
        spt_all_info = mtype_tasks[user_id][0]
        qry_all_info = mtype_tasks[user_id][1]

        spt_trajs = spt_all_info[0]
        qry_trajs = qry_all_info[0]

        task_info = []
        for target_traj in spt_trajs:
            # spt_sameo_info = []
            task_spt_info = []
            same_o_trajs, same_o_times, same_o_lengths = get_sameo_data(user_id, target_traj, sameo_user, sameo_trajs, sameo_times, sameo_lengths)
            # spt_sameo_info.append(same_o_trajs)
            # spt_sameo_info.append(same_o_times)
            # spt_sameo_info.append(same_o_lengths)
            # task_spt_info.append(spt_sameo_info)
            task_spt_info.append(same_o_trajs)
            task_spt_info.append(same_o_times)
            task_spt_info.append(same_o_lengths)
        task_info.append(task_spt_info)

        for target_traj in qry_trajs:
            # qry_sameo_info = []
            task_qry_info = []
            same_o_trajs, same_o_times, same_o_lengths = get_sameo_data(user_id, target_traj, sameo_user, sameo_trajs, sameo_times, sameo_lengths)
            # qry_sameo_info.append(same_o_trajs)
            # qry_sameo_info.append(same_o_times)
            # qry_sameo_info.append(same_o_lengths)
            # task_qry_info.append(qry_sameo_info)
            task_qry_info.append(same_o_trajs)
            task_qry_info.append(same_o_times)
            task_qry_info.append(same_o_lengths)
        task_info.append(task_qry_info)

        sameo_tasks_data[str(user_id)] = task_info

    return sameo_tasks_data


if __name__ == '__main__':
    start = time.time()

    sameo_user, sameo_trajs, sameo_times, sameo_lengths = read_sameo_data()

    mtrain_tasks = pickle.load(open(final_path + "mtrain_tasks.pkl", 'rb'))
    mvalid_tasks = pickle.load(open(final_path + "mvalid_tasks.pkl", 'rb'))
    mtest_tasks = pickle.load(open(final_path + "mtest_tasks.pkl", 'rb'))

    # generate same od data for three types of tasks
    sameo_mtrain_tasks = get_task_data(mtrain_tasks, sameo_user, sameo_trajs, sameo_times, sameo_lengths)
    print("get sameo mtrain tasks data done.")
    sameo_mvalid_tasks = get_task_data(mvalid_tasks, sameo_user, sameo_trajs, sameo_times, sameo_lengths)
    print("get sameo mvalid tasks data done.")
    sameo_mtest_tasks = get_task_data(mtest_tasks, sameo_user, sameo_trajs, sameo_times, sameo_lengths)
    print("get sameo mtest tasks data done.")

    pickle.dump(sameo_mtrain_tasks, open(final_path + "sameo_mtrain_tasks.pkl", 'wb'), protocol=4)
    print("write sameo mtrain tasks data done.")
    pickle.dump(sameo_mvalid_tasks, open(final_path + "sameo_mvalid_tasks.pkl", 'wb'), protocol=4)
    print("write sameo mvalid tasks data done.")
    pickle.dump(sameo_mtest_tasks, open(final_path + "sameo_mtest_tasks.pkl", 'wb'), protocol=4)
    print("write sameo mtest tasks data done.")

    print("write sameo_train/valid/test_tasks.pkl finished.")

    end = time.time()
    print("Step 4 finished, cost time:", (end - start) / 60, "min")