#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Generate the augmented trajectories for traj-level contrastive learning.
Input: 'mtrain_tasks.pkl', 'mvalid_tasks.pkl', 'mtest_tasks.pkl'
Format:
"mtrain_tasks.pkl": user id -> [traj1, traj2, ..., trajn]
"mtrain_tasks_aug.pkl": "mask": user id -> [masked traj1, masked traj2, ..., masked trajn]
                        "truncate": user id -> [truncated traj1, truncated traj2, ..., truncated trajn]
                        "2hop": user id -> [2hop traj1, 2hop traj2, ..., 2hop trajn]
                        "detour": user id -> [detoured traj1, detoured traj2, ..., detoured trajn]
Note:
1. all augmented trajs are regarded as positive samples.
2. traj augmentation is performed after traj padding (in Step 3)
3. all augmentation operations are performed based on the padded grid representation trajectory
'''

import time
import json
import random
import numpy as np

'''
# non_zero_len: the length of non_zero traj points
# traj_type: spt traj (keep the last item), qry traj (keep the last two items)
'''

root_path = '../data/'
hisdataset_path = root_path + 'his_porto/'


# mask: randomly mask the sequence by a given ratio
def mask(traj, non_zero_len, mask_ratio, time_dis_info, traj_type):
    if traj_type == "spt":
        mask_num = int((non_zero_len - 1) * mask_ratio)
        mask_idxs = random.sample(range(non_zero_len - 1), mask_num)
    else:
        mask_num = int((non_zero_len - 2) * mask_ratio)
        mask_idxs = random.sample(range(non_zero_len - 2), mask_num)

    masked_traj = [0 if i in mask_idxs else item for i, item in enumerate(traj)]

    masked_time_dis_info = []
    for info in time_dis_info:
        masked_info = [0 if i in mask_idxs else item for i, item in enumerate(info)]
        masked_time_dis_info.append(masked_info)

    return masked_traj, masked_time_dis_info


# truncate: cut a prefix from a sequence by a given truncate ratio and pad it with zeros:
def truncate(traj, non_zero_len, truncate_ratio, time_dis_info, traj_type):
    if traj_type == "spt":
        truncate_len = int((non_zero_len-1) * truncate_ratio)
    else:
        truncate_len = int((non_zero_len-2) * truncate_ratio)

    truncated_traj = [0] * truncate_len + traj[truncate_len:]

    truncated_time_dis_info = []
    for info in time_dis_info:
        info = info.tolist()
        if truncate_len == 0:
            truncated_info = info
        else:
            truncated_info = [0] * truncate_len + info[truncate_len:]
        truncated_time_dis_info.append(truncated_info)

    return truncated_traj, truncated_time_dis_info


# 2hop: generate the 2-hop sampling sub-trajectories
# This function generate two augmented sub-trajectories
def two_hop(traj, non_zero_len, time_up, time_low, time_up_diff, time_low_diff, dis_up, dis_low, dis_up_diff, dis_low_diff, traj_type):
    all_info = [traj, time_up, time_low, time_up_diff, time_low_diff, dis_up, dis_low, dis_up_diff, dis_low_diff]

    two_hopped_all_info1 = []
    two_hopped_all_info2 = []

    sub_infos1 = []
    sub_infos2 = []

    for info in all_info:

        if traj_type == "spt":
            sample_info = info[:non_zero_len-1]
        else:
            sample_info = info[:non_zero_len-2]

        # even number of items
        if len(sample_info) % 2 == 0:
            sub_info1 = [sample_info[i] if i % 2 == 0 else 0 for i in range(len(sample_info))]
            sub_info2 = [0 if i % 2 == 0 else sample_info[i] for i in range(len(sample_info))]
        # odd number of items
        else:
            sub_info1 = [sample_info[i] if i % 2 == 0 else 0 for i in range(len(sample_info) - 1)]
            sub_info2 = [0 if i % 2 == 0 else sample_info[i] for i in range(len(sample_info) - 1)]
            sub_info1.append(sample_info[-1])

        if traj_type == "spt":
            kept_info = info[non_zero_len-1:]
        else:
            kept_info = info[non_zero_len-2:]

        sub_info1.extend(kept_info)
        sub_info2.extend(kept_info)

        sub_infos1.append(sub_info1)
        sub_infos2.append(sub_info2)

        # two_hopped_all_info1.append(sub_infos1)
        # two_hopped_all_info2.append(sub_infos2)
    # return two_hopped_all_info1, two_hopped_all_info2
    return sub_infos1, sub_infos2


# detour: randomly select a short part of a trajectory and replace it with another path
# with the same origin and destination to generate a detour trajectory
# replace the target traj with a sub-sequence in replace_traj
def detour(target_traj, non_zero_len, replace_traj, replace_ratio, traj_type):
    replace_num = int(replace_ratio * non_zero_len)

    if len(replace_traj) != 0:
        if replace_num % 2 == 0:
            start_idx = (len(replace_traj) - replace_num) // 2
        else:
            start_idx = (len(replace_traj) - replace_num) // 2 + 1

        a_idx = (len(target_traj) - non_zero_len) // 2
        b_idx = start_idx

        # store the last one(spt)/two(qry) non-zero elements of target_traj
        if traj_type == "spt":
            kept_elements = target_traj[-1:]
        else:
            kept_elements = target_traj[-2:]

        for _ in range(replace_num):
            while target_traj[a_idx] == 0:
                a_idx += 1
            target_traj[a_idx] = replace_traj[b_idx]
            a_idx += 1
            b_idx += 1
        if traj_type == "spt":
            target_traj[-1:] = kept_elements
        else:
            target_traj[-2:] = kept_elements

    return target_traj


def generate_augmented_traj(traj_info, mask_bool=True, truncate_bool=True, two_hop_bool=True, detour_bool=True,
                            mask_ratio=0.2, truncate_ratio=0.3, replace_ratio=0.3, traj_type='spt', user_id_data=None):

    traj, truth, len_pad, length, time_info, dis_info = traj_info
    next_truth, des_truth = truth
    time_up, time_low, time_up_diff, time_low_diff = time_info
    dis_up, dis_low, dis_up_diff, dis_low_diff = dis_info

    spt_user_id_data, qry_user_id_data = user_id_data

    masked_info = []
    truncated_info = []
    two_hopped_info = []
    detoured_info = []

    if mask_bool:
        masked_traj, masked_time_dis_info = mask(traj, length, mask_ratio, [time_up, time_low, time_up_diff, time_low_diff, dis_up, dis_low, dis_up_diff, dis_low_diff], traj_type)
        masked_info = [masked_traj] + masked_time_dis_info

    if truncate_bool:
        truncated_traj, truncated_time_dis_info = truncate(traj, length, truncate_ratio, [time_up, time_low, time_up_diff, time_low_diff, dis_up, dis_low, dis_up_diff, dis_low_diff], traj_type)
        truncated_info = [truncated_traj] + truncated_time_dis_info

    if two_hop_bool:
        two_hopped_info = two_hop(traj, length, time_up, time_low, time_up_diff, time_low_diff, dis_up, dis_low, dis_up_diff, dis_low_diff, traj_type)

    if detour_bool:
        if traj_type == 'spt':
            user_sameo_trajs = spt_user_id_data[0]
        else:
            user_sameo_trajs = qry_user_id_data[0]

        replace_traj = random.choice(user_sameo_trajs)
        detoured_traj = detour(traj, length, replace_traj, replace_ratio, traj_type)
        detoured_info = [detoured_traj] + [time_up, time_low, time_up_diff, time_low_diff, dis_up, dis_low, dis_up_diff, dis_low_diff]

    return masked_info, truncated_info, two_hopped_info, detoured_info  #, next_truth, des_truth, len_pad, length


# Output format:
# spt_traj_i, spt_next_truth_i, spt_des_truth_i, spt_len_pad_i, spt_length_i, \
# spt_time_up_i, spt_time_low_i, spt_time_up_diff_i, spt_time_low_diff_i, \
# spt_dis_up_i, spt_dis_low_i, spt_dis_up_diff_i, spt_dis_low_diff_i
# order by [masked traj info, truncated traj info, two_hopped traj info, detoured traj info]
def combine_all_augmented_info(traj_info, masked_info, truncated_info, two_hopped_info, detoured_info):
    next_truth = traj_info[1][0]
    des_truth = traj_info[1][1]
    len_pad = traj_info[2]
    length = traj_info[3]

    combined_augmented_traj_info = []

    # combined_augmented_traj_info.append(original_traj_info)

    all_info = [masked_info, truncated_info, two_hopped_info[0], two_hopped_info[1], detoured_info]

    for info in all_info:
        # new_augmented_traj_info = []
        traj_info = info[:1]
        time_info = info[1:5]
        np_time_info = [np.array(info) for info in time_info]
        dis_info = info[5:]
        np_dis_info = [np.array(info) for info in dis_info]
        new_augmented_traj_info = [traj_info[0], [next_truth, des_truth], len_pad, length, np_time_info, np_dis_info]

        combined_augmented_traj_info.append(new_augmented_traj_info)

    return combined_augmented_traj_info













