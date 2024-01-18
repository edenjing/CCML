#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import pickle
import numpy as np
from model.pre_model import *


class Learner(nn.Module):
    '''
    Base Model in CMCTP
    '''
    def __init__(self, config, device):
        super(Learner, self).__init__()

        self.device = device
        # self.uodemb = uemb
        self.pre_model = TrajPreLocalAttnLong(config['hidden_neurons'], device=self.device)
        # self.linear1_next = torch.nn.Linear(config['hidden_neurons']+10, 128)  # concatenate layer
        self.linear1_next = torch.nn.Linear(config['hidden_neurons'], 128)  # concatenate layer
        self.linear2_next = torch.nn.Linear(128, 64)  # fully-connected network
        self.next_linear = torch.nn.Linear(64, 200*200+1)  # prediction layer
        self.activation_Relu = torch.nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.3)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)

    # spt_traj, spt_len_pad, spt_length, spt_time_up, spt_time_low, spt_time_up_diff, spt_time_low_diff, spt_dis_up, spt_dis_low, spt_dis_up_diff, spt_dis_low_diff,
    def forward(self, cur, c_length, c_traj_len, c_tu, c_tl, c_tu_slot, c_tl_slot, c_su, c_sl, c_su_slot, c_sl_slot):
        # ud = ud.float()
        # emb_uod = self.uodemb(ud)

        next_value = self.pre_model(cur, c_length, c_traj_len, c_tu, c_tl, c_tu_slot, c_tl_slot, c_su, c_sl, c_su_slot, c_sl_slot) # emb_traj, emb_tu, emb_tl, emb_tu_slot, emb_tl_slot, emb_su, emb_sl, emb_su_slot, emb_sl_slot, lennew, len1, length, record_len)
        # torch.cat(tensors,dim=0,*,out=None): concatenate the given sequence of seq tensors in the given dimension.
        # trajs = torch.cat((next_value, emb_uod), -1)

        # if len(x2.shape) == 1:
        #     concate_next = torch.cat((next_value, emb_uod), 0)
        # else:
        #     concate_next = torch.cat((next_value, emb_uod), 1)
        concate_next = next_value

        concate_next1 = self.dropout(concate_next)
        linear_out = self.linear1_next(concate_next1)
        linear_out1 = self.linear2_next(linear_out)
        linear_out2 = self.next_linear(linear_out1)
        next_out1 = F.log_softmax(linear_out2, dim=-1)

        # next_out1: predicted result
        # next_value: output representation of current traj
        return next_out1, next_value









