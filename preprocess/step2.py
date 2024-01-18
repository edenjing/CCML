#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Step 2: split trajs/times/dis for each train/valid/target user.
1. select TRAIN_TRAJS_NUM/TEST_TRAJS_NUM trajs/times/dis for train/valid/target users
2. sort trajs for each user and ensure the lengths of all trajs/times/dis must less than MAX_TRAJ_LEN
'''
from constants import *
import os
import time

# "data"/dataname = "trajs"/"times"/"dis"
def read_raw_data(split_path, dataname):
    max_len_of_data = 0
    min_len_of_data = 1000000

    types = ["train", "valid", "test"]
    data_by_userid = {type_: {} for type_ in types}
    for i, type_ in enumerate(types):
        user_data_dir = os.path.join(split_path + "{}_user_{}/".format(types[i], dataname))
        for fname in os.listdir(user_data_dir):
            if fname.endswith(".txt"):
                user_id = fname[:-4]
                user_data = []
                with open(os.path.join(user_data_dir, fname), "r") as f:
                    for line in f:
                        data = [int(p) for p in line.strip().split(",")]
                        user_data.append(data)
                        max_len_of_data = max(max_len_of_data, len(data))
                        min_len_of_data = min(min_len_of_data, len(data))
                data_by_userid[type_][user_id] = user_data
    print("max_len_of_{}: ".format(dataname), max_len_of_data, "\tmin_len_of_{}: ".format(dataname), min_len_of_data)
    return data_by_userid


def get_traj_num(type):
    if type == "train":
        return TRAIN_TRAJS_NUM
    else:
        return TEST_TRAJS_NUM


def select_data(split_user_datasets, dataname):
    for type, userid_datasets in split_user_datasets.items():
        selected_user_datasets_dir = split_path + "selected_{}_user_{}/".format(type, dataname)

        if not os.path.exists(selected_user_datasets_dir):
            os.mkdir(selected_user_datasets_dir)

        for userid, datasets in userid_datasets.items():
            # sort the data for each user by the length decreasing order
            sorted_datasets = sorted(datasets, key=lambda data: len(data), reverse=True)
            with open(selected_user_datasets_dir + userid + ".txt", "w") as f:
                restricted_data_num = get_traj_num(type)
                i = 0
                for data in sorted_datasets:
                    # if the number of trajs of a user is more than restricted_traj_num,
                    # only select restricted_traj_num trajs
                    if i >= restricted_data_num:
                        break
                    # the length of traj must be larger than MIN_TRAJ_LEN,
                    # and if the length of traj is larger than MAX_TRAJ_LEN, select the last part with MAX_TRAJ_LEN
                    if len(data) < MIN_TRAJ_LEN:
                        continue
                    if len(data) > MAX_TRAJ_LEN:
                        data = data[-MAX_TRAJ_LEN:]
                    i += 1
                    f.write(",".join(str(p) for p in data) + '\n')
        print("store selected_{}_user_{} by user id finished.".format(type, dataname))

if __name__ == '__main__':
    start = time.time()
    datanames = ["trajs", "times", "dis"]
    for dataname in datanames:
        split_user_datasets = read_raw_data(split_path, dataname)
        select_data(split_user_datasets, dataname)

    end = time.time()
    print("Step 2 finished, cost time:", (end - start) / 60, "min")