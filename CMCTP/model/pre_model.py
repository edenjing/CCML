#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.STLSTM import *


class TrajPreLocalAttnLong(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, hidden_neurons, device):
        super(TrajPreLocalAttnLong, self).__init__()
        self.loc_size = 200 * 200 + 1
        self.loc_emb_size = 100
        self.hidden_size = hidden_neurons
        self.device = device

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)  # trajectory location information
        self.embedding_tslot = torch.nn.Embedding(num_embeddings=92, embedding_dim=12)  # time interval information
        self.embedding_sslot = torch.nn.Embedding(num_embeddings=92, embedding_dim=12)  # distance interval information
        # self.fc_attn = nn.Linear(input_size, self.hidden_size)
        self.rnn_decoder = STLSTM(config_settings['hidden_neurons'], config_settings['hidden_neurons'],
                                  config_settings['d'], config_settings['d1'])

        self.dropout = nn.Dropout(p=0.3)
        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)  # Fills the input Tensor with a (semi) orthogonal matrix # 用(半)正交矩阵填充输入张量
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, traj, lennew, traj_len, tu, tl, tu_slot, tl_slot, su, sl, su_slot, sl_slot):
        tu = tu.long()
        tl = tl.long()
        tu_slot = tu_slot.long()
        tl_slot = tl_slot.long()
        su = su.long()
        sl = sl.long()
        su_slot = su_slot.long()
        sl_slot = sl_slot.long()

        emb_tu = self.embedding_tslot(tu)
        emb_tl = self.embedding_tslot(tl)
        emb_su = self.embedding_sslot(su)
        emb_sl = self.embedding_sslot(sl)
        loc_emb = self.emb_loc(traj)

        x = self.dropout(loc_emb)

        lstm_outs_h = []
        traj_len = traj_len.long()
        # originally, more than one trajectory
        # traj_leni = int(max(traj_len.item()))
        # print("traj_len: ", traj_len)

        traj_leni = traj_len.item()
        for t in range(traj_leni + 1):  # dataloader
            inputsi = x[:, t]  # [0][t]
            input_lengthi = lennew[:, t]
            t_upi = emb_tu[:, t]
            t_lowi = emb_tl[:, t]
            t_up_diffi = tu_slot[:, t]
            t_low_diffi = tl_slot[:, t]

            s_upi = emb_su[:, t]
            s_lowi = emb_sl[:, t]
            s_up_diffi = su_slot[:, t]
            s_low_diffi = sl_slot[:, t]
            bs, es = inputsi.shape
            if t == 0:
                h = Variable(torch.zeros(bs, self.hidden_size), requires_grad=True).to(self.device)
                c = Variable(torch.zeros(bs, self.hidden_size), requires_grad=True).to(self.device)
                (h, c) = self.rnn_decoder(inputsi, 0, 0, t_upi, t_lowi, t_up_diffi, t_low_diffi, s_upi, s_lowi,
                                          s_up_diffi, s_low_diffi, input_lengthi, h, c, predict=0)
                lstm_outs_h.append(h)
            else:
                (h, c) = self.rnn_decoder(inputsi, 0, 0, t_upi, t_lowi, t_up_diffi, t_low_diffi, s_upi, s_lowi,
                                          s_up_diffi, s_low_diffi, input_lengthi, h, c, predict=0)
                lstm_outs_h.append(h)

        # torch.stack(tensors, dim=0, *, out=None): Concatenate a sequence of tensors along a new dimension.
        # torch.transpose(input, dim0, dim1): The given dimensions dim0 and dim1 are swapped.
        lstm_outs_h1 = torch.stack(lstm_outs_h)  # 看下怎么stack的？
        finalout = torch.transpose(lstm_outs_h1, 1, 0)

        h_new = []
        for i in range(traj.shape[0]):
            h_new.append(finalout[i][traj_len[i]])

        hidden_state = torch.stack(h_new)

        return hidden_state