#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：CMCTP 
@File    ：constants.py
@Author  ：Eden
@Date    ：2023/4/18 1:02 下午 
'''

#######
# TODO: you may decide your setting.

root_path = '../data/'
# hisdataset_path = root_path + 'his_porto/'
hisdataset_path = root_path + 'his_beijing/'

newdataset_path = root_path + 'dataset/'
split_path = newdataset_path + 'split/'
config_path = newdataset_path + 'config/'
final_path = newdataset_path + 'final/'


# for users
TRAIN_USERS_NUM = 250
VALID_USERS_NUM = 84
TEST_USERS_NUM = 100

# for trajs in train/valid/test dataset
TRAIN_TRAJS_NUM = 1800
TEST_TRAJS_NUM = 500  # also for valid

# for each traj
MAX_TRAJ_LEN = 100
MIN_TRAJ_LEN = 10

# support and query set split ratio
S_Q_RATIO = 0.6


def get_users(which="base"):
    users_file = config_path + which + "_users.txt"
    users = []
    with open(users_file, 'r') as f:
        for line in f:
            user = line.strip()
            users.append(user)
    return users

