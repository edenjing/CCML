#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Step 3: make the final dataset files for meta-learning process.
output: 'mtrain_tasks.pkl' 'mvalid_tasks.pkl' 'mtest_tasks.pkl',
with each file stores a list, each element of the list is the data (all information) of a meta-training user/task

1. pad traj set and time/dis diff sets for train/valid/test users
2. combine all information data by type_userid_{spt, qry}  
3. write all information into pickle files
'''
import os
import pickle
import time
import numpy as np
from constants import *


# Here can also read the raw data before Step 2
# dataname = "trajs"/"times"/"dis"
def read_selected_data(split_path, dataname):
    types = ["train", "valid", "test"]
    datasets_by_userid = {type_: {} for type_ in types}
    for i, type_ in enumerate(types):
        user_datasets_dir = os.path.join(split_path + "selected_{}_user_{}/".format(type_, dataname))
        for fname in os.listdir(user_datasets_dir):
            if fname.endswith(".txt"):
                user_id = fname[:-4]
                user_datasets = []
                with open(os.path.join(user_datasets_dir, fname), "r") as f:
                    for line in f:
                        user_dataset = [int(p) for p in line.strip().split(",")]
                        user_datasets.append(user_dataset)
                datasets_by_userid[type_][user_id] = user_datasets
    return datasets_by_userid


# pad trajs (with 0) for users
def pad_trajs(type, type_data):  # type_data: dict{user_id, trajs}
    print("padding data for users in selected_{}_user_trajs directory...".format(type))
    type_padded_data = {}
    for user_id, user_trajs in type_data.items():
        # for one suer
        padded_trajs = []  # store trajs after padding
        labels = []  # destination cell id
        next_labels = []  # the next one's cell id
        # next_labels_one = np.zeros((len(user_trajs), 5))  # shape: (len(user_trajs), 5)
        next_labels_two = np.zeros(len(user_trajs))  # shape: len(user_trajs)
        trajs_len = []
        trajs_len_one = []
        length = np.zeros((len(user_trajs), MAX_TRAJ_LEN, 1))

        # generate support set
        spt_padded_trajs = []
        spt_labels = []
        spt_next_labels = []
        # spt_next_labels_one = np.zeros((len(user_trajs), 5))  # shape: (len(user_trajs), 5)
        spt_next_labels_two = np.zeros(len(user_trajs))  # shape: len(user_trajs)

        spt_trajs_len = []
        spt_trajs_len_one = []
        spt_length = np.zeros((len(user_trajs), MAX_TRAJ_LEN, 1))

        # break(split) the raw traj
        max_bre_idx = 0
        for i, traj in enumerate(user_trajs):
            idx = len(traj) - 1
            if int(idx * S_Q_RATIO) + 1 < MAX_TRAJ_LEN:
                bre_idx = int(idx * S_Q_RATIO) + 1
            else:
                bre_idx = int(idx * S_Q_RATIO)
            max_bre_idx = max(max_bre_idx, bre_idx)
            bre_traj = traj[:bre_idx]

            spt_bre_traj = traj[:bre_idx-1]   # support set

            while len(bre_traj) < MAX_TRAJ_LEN:
                bre_traj.append(0)
            while len(spt_bre_traj) < MAX_TRAJ_LEN:  # support set
                spt_bre_traj.append(0)

            labels.append(traj[idx])   # destination cell id
            spt_labels.append(traj[idx-1])  # support set

            for j in range(bre_idx+1):
                length[i][j][0] = 1
            for k in range(bre_idx+1-1):  # support set
                spt_length[i][k][0] = 1

            next_labels_two[i] = traj[bre_idx]
            spt_next_labels_two[i] = traj[bre_idx-1]  # support set

            next_labels.append(traj[bre_idx])
            padded_trajs.append(bre_traj)
            trajs_len.append(idx)
            trajs_len_one.append(bre_idx-1)

            # support set
            spt_next_labels.append(traj[bre_idx-1])
            spt_padded_trajs.append(spt_bre_traj)
            spt_trajs_len.append(idx-1)
            spt_trajs_len_one.append(bre_idx-2)

        groundtruth = [next_labels_two, labels]
        spt_groundtruth = [spt_next_labels_two, spt_labels]

        # add into type_tasks_data
        user_padded_data = [padded_trajs, groundtruth, length, trajs_len_one, spt_padded_trajs, spt_groundtruth, spt_length, spt_trajs_len_one]
        type_padded_data[user_id] = user_padded_data
    return type_padded_data


# Return: qry_traj, qry_gd, qry_length[1], qry_len[int], spt_traj, spt_gd, spt_length[1], spt_len[int]
def get_padded_trajs(trajs_by_userid):
    all_padded_trajs = {}
    for type, type_data in trajs_by_userid.items():
        padded_data = pad_trajs(type, type_data)  # padded_data is dictionary
        all_padded_trajs[type] = padded_data
    return all_padded_trajs


# compute the corresponding up/low/up_diff/low_diff for each slot_data (2D list)
# time_slot: divided by 60s
# Example: time interval 80s
# up: 80 // 60 + 1; up_diff: 60 * 2 - 80
# low: 80 // 60; lower_diff: 80 - 60
def compute_up_low_diff(slot_data, dataname):
    up_2d_list = []
    low_2d_list = []
    up_diff_2d_list = []
    low_diff_2d_list = []

    for slot_list in slot_data:
        up_list = []
        low_list = []
        up_diff_list = []
        low_diff_list = []
        for slot in slot_list:
            if dataname == "times":
                up = slot // 60 + 1
                low = slot // 60
                up_diff = 60 * up - slot
                low_diff = slot - 60 * low
            else:    # dataname = "dis"
                up = slot // 100 + 1
                low = slot // 100
                up_diff = 100 * up - slot
                low_diff = slot - 100 * low
            up_list.append(up)
            low_list.append(low)
            up_diff_list.append(up_diff)
            low_diff_list.append(low_diff)
        up_2d_list.append(up_list)
        low_2d_list.append(low_list)
        up_diff_2d_list.append(up_diff_list)
        low_diff_2d_list.append(low_diff_list)
    return up_2d_list, low_2d_list, up_diff_2d_list, low_diff_2d_list


# spt prefix: support set
# no spt prefix: query set
# all converted into ndarray
def pad_times_or_dis(type, slot_datasets, dataname):
    print("padding data for users in selected_{}_user_{} directory...".format(type, dataname))
    start = time.time()
    type_padded_data = {}
    for user_id, slot_data in slot_datasets.items():
        # for one suer
        up, low, up_diff, low_diff = compute_up_low_diff(slot_data, dataname)

        # up = slot_data[0]
        # low = slot_data[1]
        # up_diff = slot_data[2]
        # low_diff = slot_data[3]

        up_percent = []
        low_percent = []
        up_diff_percent = []
        low_diff_percent = []

        # support set
        spt_up_percent = []
        spt_low_percent = []
        spt_up_diff_percent = []
        spt_low_diff_percent = []

        for h in range(len(up)):
            idx = len(up[h]) - 1
            if int(idx * S_Q_RATIO) + 1 < MAX_TRAJ_LEN:
                bre_idx = int(idx * S_Q_RATIO) + 1
            else:
                bre_idx = int(idx * S_Q_RATIO)
            upi = list(up[h][0:bre_idx])
            lowi = list(low[h][0:bre_idx])
            up_diffi = list(up_diff[h][0:bre_idx])
            low_diffi = list(low_diff[h][0:bre_idx])

            while len(upi) < MAX_TRAJ_LEN:
                upi.append(0)
                lowi.append(0)
                up_diffi.append(0)
                low_diffi.append(0)

            up_percent.append(upi)
            low_percent.append(lowi)
            up_diff_percent.append(up_diffi)
            low_diff_percent.append(low_diffi)

            # support set
            spt_upi = list(up[h][0:bre_idx-1])
            spt_lowi = list(low[h][0:bre_idx-1])
            spt_up_diffi = list(up_diff[h][0:bre_idx-1])
            spt_low_diffi = list(low_diff[h][0:bre_idx-1])

            while len(spt_upi) < MAX_TRAJ_LEN:
                spt_upi.append(0)
                spt_lowi.append(0)
                spt_up_diffi.append(0)
                spt_low_diffi.append(0)

            spt_up_percent.append(spt_upi)
            spt_low_percent.append(spt_lowi)
            spt_up_diff_percent.append(spt_up_diffi)
            spt_low_diff_percent.append(spt_low_diffi)

        np_up_percent = np.array(up_percent)
        np_low_percent = np.array(low_percent)
        np_up_diff_percent = np.array(up_diff_percent)
        np_low_diff_percent = np.array(low_diff_percent)

        spt_np_up_percent = np.array(spt_up_percent)
        spt_np_low_percent = np.array(spt_low_percent)
        spt_np_up_diff_percent = np.array(spt_up_diff_percent)
        spt_np_low_diff_percent = np.array(spt_low_diff_percent)

        # for one user
        user_padded_data = [np_up_percent, np_low_percent, np_up_diff_percent, np_low_diff_percent,
                            spt_np_up_percent, spt_np_low_percent, spt_np_up_diff_percent, spt_np_low_diff_percent]
        type_padded_data[user_id] = user_padded_data
    end = time.time()
    print("pad {} finished, cost time:".format(dataname), (end - start) / 60, "min")
    # for one type, one dataname
    return type_padded_data


# for one dataname
# Return: qry_up, qry_low, qry_up_diff, qry_low_diff, spt_up, spt_low, spt_up_diff, spt_low_diff
def get_padded_times_or_dis(times_or_dis_by_userid, dataname):
    all_padded_datasets = {}
    for type, type_data in times_or_dis_by_userid.items():
        padded_dataset = pad_times_or_dis(type, type_data, dataname)  # padded_data is dictionary
        all_padded_datasets[type] = padded_dataset
    return all_padded_datasets


# read and pad users_trajs/times/dis in three types
def get_padded_data():
    padded_trajs = {}
    padded_times = {}
    padded_dis = {}
    datanames = ["trajs", "times", "dis"]
    for dataname in datanames:
        # read data
        datasets_by_userid = read_selected_data(split_path, dataname)
        # pad data
        if dataname == "trajs":
            padded_trajs = get_padded_trajs(datasets_by_userid)
            print("get padded trajs finished.")
        elif dataname == "times":
            padded_times = get_padded_times_or_dis(datasets_by_userid, dataname)
            print("get padded times finished.")
        else:
            padded_dis = get_padded_times_or_dis(datasets_by_userid, dataname)
            print("get padded dis finished.")
    print("get padded trajs/times/dis data finished.")
    return padded_trajs, padded_times, padded_dis


# for "times" and "dis"
# Return: [0]:[qry_up, qry_low, qry_up_diff, qry_low_diff]
def combine_padded_times_or_dis(padded_data):
    # divide into spt and qry
    # spt prefix: support set
    # no spt prefix: query set
    type_userid_diff = {}
    for type, user_padded_data in padded_data.items():
        # user id: [up, low, up_diff, low_diff, spt_up, spt_low, spt_up_diff, spt_low_diff]
        userid_diff = {}
        for userid, padded_data in user_padded_data.items():
            qry_diff = padded_data[:4]
            spt_diff = padded_data[4:]
            userid_diff[userid] = [qry_diff, spt_diff]
        type_userid_diff[type] = userid_diff
    return type_userid_diff
    print("combine padded times/dis finished.")


def combine_user_all_information(padded_trajs, padded_times, padded_dis):
    # user id: [padded_trajs, groundtruth, length, trajs_len_one,
    # spt_padded_trajs, spt_groundtruth, spt_length, spt_trajs_len_one]
    combined_padded_times = combine_padded_times_or_dis(padded_times)
    combined_padded_dis = combine_padded_times_or_dis(padded_dis)

    type_userid_qry_all_information = {}
    type_userid_spt_all_information = {}

    # type: "train"/"valid"/"test"
    for type, type_info in padded_trajs.items():
        userid_qry_all_information = {}
        userid_spt_all_information = {}

        for user_id, traj_info in type_info.items():
            # qry first, spt second
            qry_all_information = [traj_info[0], traj_info[1], traj_info[2], traj_info[3],
                                combined_padded_times[type][user_id][0], combined_padded_dis[type][user_id][0]]
            # user id: [up, low, up_diff, low_diff, spt_up, spt_low, spt_up_diff, spt_low_diff]
            spt_all_information = [traj_info[4], traj_info[5], traj_info[6], traj_info[7],
                                combined_padded_times[type][user_id][1], combined_padded_dis[type][user_id][1]]

            userid_qry_all_information[user_id] = qry_all_information
            userid_spt_all_information[user_id] = spt_all_information

        type_userid_qry_all_information[type] = userid_qry_all_information
        type_userid_spt_all_information[type] = userid_spt_all_information
    # return qry and spt all information of three types
    # format: train/valid/test -> userid -> qry/spt_all_information
    return type_userid_qry_all_information, type_userid_spt_all_information


# store into pickle files by type
def write_user_all_information(type_userid_spt_all_information, type_userid_qry_all_information):
    types = ["train", "valid", "test"]
    type_all_information_dir = final_path
    if not os.path.exists(type_all_information_dir):
        os.mkdir(type_all_information_dir)

    for type in types:
        userid_all_information = {}

        userid_spt_all_information = type_userid_spt_all_information[type]
        userid_qry_all_information = type_userid_qry_all_information[type]

        for userid, spt_all_information in userid_spt_all_information.items():
            # order by: userid: spt + qry
            userid_spt_qry_all_information = [spt_all_information, userid_qry_all_information[userid]]
            userid_all_information[userid] = userid_spt_qry_all_information

        pickle.dump(userid_all_information, open(type_all_information_dir + "m{}_tasks.pkl".format(type), 'wb'), protocol=4)
        print("write m{}_tasks.pkl finished.".format(type))

    print("write pickles finished.")


if __name__ == '__main__':
    start = time.time()
    padded_trajs, padded_times, padded_dis = get_padded_data()
    # combined_padded_times, combined_padded_dis = combine_padded_data(padded_times, padded_dis)
    type_userid_qry_all_information, type_userid_spt_all_information = combine_user_all_information(padded_trajs, padded_times, padded_dis)
    write_user_all_information(type_userid_spt_all_information, type_userid_qry_all_information)
    end = time.time()
    print("Step 3 finished, cost time:", (end - start) / 60, "min")