#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Step 1: sort users by the number of their historical trajectories and split into base/valid/test users,
and store train/valid/test_user_trajs/times/dis by user id.
'''
import json
from constants import *
import os
import time


def read_traj_time_dis():  # 435 users
    with open(hisdataset_path + 'users_trajs.json', 'r') as f1:
        user_trajs = json.load(f1)

    # Porto: users_time.json
    # Beijing: users_times.json
    with open(hisdataset_path + 'users_time.json', 'r') as f2:
        user_times = json.load(f2)
    with open(hisdataset_path + 'users_length.json', 'r') as f3:
        user_dis = json.load(f3)
    #     user_dis = []
    return user_trajs, user_times, user_dis


def sort_by_trajnum(user_data):
    sorted_user_data = dict(sorted(user_data.items(), key=lambda item: -len(item[1])))  # 434 users
    max_traj_num = len(next(iter(sorted_user_data.items()))[1])
    min_traj_num = len(sorted_user_data.popitem()[1])
    print("max_traj_num: ", max_traj_num, "\tmin_traj_num: ", min_traj_num)
    return sorted_user_data


def split_user_data(user_data):
    sorted_user_data = sort_by_trajnum(user_data)
    sorted_user_data = list(sorted_user_data.items())
    split_ratio = [TRAIN_USERS_NUM, VALID_USERS_NUM, TEST_USERS_NUM]
    start_index = 0
    splitted_user_data = []
    for r in split_ratio:
        sub_user_data = dict(sorted_user_data[start_index:start_index + r])
        splitted_user_data.append(sub_user_data)
        start_index += r

    # store the user ids into txt file
    if not os.path.exists(config_path):
        os.mkdir(config_path)
    filenames = ["base_users.txt", "valid_users.txt", "target_users.txt"]
    for i, sub_user_data in enumerate(splitted_user_data):
        with open(config_path + filenames[i], "w") as f:
            f.write("\n".join(sub_user_data.keys()))
    print("split user data finished.")
    return splitted_user_data


# store the train/valid/test data(traj, time, dis) for each user
def store_data_by_userid(split_user_data, dataname):
    types = ["train", "valid", "test"]
    for i, sub_user_data in enumerate(split_user_data):
        user_data_dir = split_path + "{}_user_{}/".format(types[i], dataname)
        # print(types[i], len(sub_user_trajs))
        if not os.path.exists(user_data_dir):
            os.mkdir(user_data_dir)
        for user_id, datasets in sub_user_data.items():
            with open(user_data_dir + user_id + ".txt", "w") as f:
                for data in datasets:
                    f.write(",".join(str(p) for p in data) + '\n')
    print("store train/valid/test_user_{} by user id finished.".format(dataname))


if __name__ == '__main__':
    start = time.time()

    # sorted_user_trajs = sort_user_trajs_by_trajnum()
    # split_user_trajs = split_user_trajs(sorted_user_trajs)
    # store_trajs_by_userid(split_user_trajs)
    # user_trajs, user_times, user_dis = read_traj_time_dis()

    user_datasets = read_traj_time_dis()
    datanames = ["trajs", "times", "dis"]
    for i, user_data in enumerate(user_datasets):
        splitted_user_data = split_user_data(user_data)
        store_data_by_userid(splitted_user_data, datanames[i])

    end = time.time()
    print("Step 1 finished, cost time:", (end - start) / 60, "min")
