#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import numpy as np
from torch.utils.data import Dataset


def to_torch_long(in_list, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    return torch.LongTensor(in_list).to(device)  # 64-bit integer


def to_torch(in_list):
    return torch.from_numpy(np.array(in_list))  # Creates a Tensor from a numpy.ndarray.


# data: data[task_id]: [spt_all_info, qry_all_info]
# [spt_all_info / qry_all_info]: order by traj id: [traj, [next truth, des truth], len[pad with 1], length,
# [time_up, time_low, time_up_diff, time_low_diff], [dis_up, dis_low, dis_up_diff, dis_low_diff]]
def data_to_device(spt_qry_info, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # [traj, [next truth, des truth], len[pad with 1], length,
    # [time_up, time_low, time_up_diff, time_low_diff], [dis_up, dis_low, dis_up_diff, dis_low_diff]]
    # print(len(spt_qry_info))
    traj = to_torch_long(spt_qry_info[0]).to(device)
    next_truth = to_torch(spt_qry_info[1][0]).to(device)
    des_truth = to_torch(spt_qry_info[1][1]).to(device)
    len_pad = to_torch_long(spt_qry_info[2]).to(device)
    length = to_torch(spt_qry_info[3]).to(device)

    time_up = to_torch_long(spt_qry_info[4][0]).to(device)
    time_low = to_torch_long(spt_qry_info[4][1]).to(device)
    time_up_diff = to_torch_long(spt_qry_info[4][2]).to(device)
    time_low_diff = to_torch_long(spt_qry_info[4][3]).to(device)

    dis_up = to_torch_long(spt_qry_info[5][0]).to(device)
    dis_low = to_torch_long(spt_qry_info[5][1]).to(device)
    dis_up_diff = to_torch_long(spt_qry_info[5][2]).to(device)
    dis_low_diff = to_torch_long(spt_qry_info[5][3]).to(device)

    return traj, next_truth, des_truth, len_pad, length, time_up, time_low, time_up_diff, time_low_diff, dis_up, dis_low, dis_up_diff, dis_low_diff


class TrajDataLoader(Dataset):
    def __init__(self, traj, lennew, trajs_len, up, low, up_diff, low_diff, sup, slow, sup_diff, slow_diff, y_next, y_destination, transform=None):  #userid,
        self.traj = traj
        self.lennew = lennew
        self.trajs_len = trajs_len
        self.up = up
        self.low = low
        self.up_diff = up_diff
        self.low_diff = low_diff
        self.sup = sup
        self.slow = slow
        self.sup_diff = sup_diff
        self.slow_diff = slow_diff
        self.y_next = y_next
        self.y_destination = y_destination
        # self.user_d = user_d
        # self.sh = sh
        self.transform = transform

    def __len__(self):
        if len(self.traj.shape) == 1:
            return 1
        else:
            return len(self.y_next)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if len(self.traj.shape) > 1:
            user_trajctory = self.traj[idx]
            traj_lennew = self.lennew[idx]
            traj_len = self.trajs_len[idx]
            trajctory_time1 = self.up[idx]
            trajctory_time2 = self.low[idx]
            trajctory_time3 = self.up_diff[idx]
            trajctory_time4 = self.low_diff[idx]
            trajctory_loca1 = self.sup[idx]
            trajctory_loca2 = self.slow[idx]
            trajctory_loca3 = self.sup_diff[idx]
            trajctory_loca4 = self.slow_diff[idx]
            y_next = self.y_next[idx]
            y_destination = self.y_destination[idx]
            # user_di = self.user_d[idx]
            # shi = self.sh[idx]
        else:
            user_trajctory = self.traj
            traj_lennew = self.lennew
            traj_len = self.trajs_len
            trajctory_time1 = self.up
            trajctory_time2 = self.low
            trajctory_time3 = self.up_diff
            trajctory_time4 = self.low_diff
            trajctory_loca1 = self.sup
            trajctory_loca2 = self.slow
            trajctory_loca3 = self.sup_diff
            trajctory_loca4 = self.slow_diff
            y_next = self.y_next
            y_destination = self.y_destination
            # user_di = self.user_d
            # shi = self.sh
        # return user_trajctory, traj_lennew, trajctory_time1, trajctory_time2, trajctory_time3, trajctory_time4, trajctory_loca1, \
        #        trajctory_loca2, trajctory_loca3, trajctory_loca4, y_next, y_destination, traj_len, user_di, shi    #u_id,
        return user_trajctory, traj_lennew, traj_len, trajctory_time1, trajctory_time2, trajctory_time3, trajctory_time4, trajctory_loca1, \
               trajctory_loca2, trajctory_loca3, trajctory_loca4, y_next, y_destination

