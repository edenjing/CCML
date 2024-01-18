#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
from config import *
import math


class STLSTM(nn.Module):
    """docstring for SpatioTemporal_LSTM"""

    def __init__(self, hidden_dim, input_dim, d_dim, d1_dim):  # , nt_dim
        super(STLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.d_dim = d_dim
        self.d1_dim = d1_dim

        self.kernel_Wi = Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.kernel_Wf = Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.kernel_Wc = Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        self.kernel_Wog = Parameter(torch.Tensor(self.input_dim, self.hidden_dim))

        self.kernel_Ui = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.kernel_Uf = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.kernel_Uc = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        self.kernel_Uog = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))

        self.kernel_Ti = Parameter(torch.Tensor(self.d_dim, self.hidden_dim))
        self.kernel_Tf = Parameter(torch.Tensor(self.d_dim, self.hidden_dim))
        self.kernel_To = Parameter(torch.Tensor(self.d_dim, self.hidden_dim))

        self.kernel_Si = Parameter(torch.Tensor(self.d1_dim, self.hidden_dim))
        self.kernel_Sf = Parameter(torch.Tensor(self.d1_dim, self.hidden_dim))
        self.kernel_So = Parameter(torch.Tensor(self.d1_dim, self.hidden_dim))

        self.bias_i = Parameter(torch.Tensor(hidden_dim))
        self.bias_f = Parameter(torch.Tensor(hidden_dim))
        self.bias_c = Parameter(torch.Tensor(hidden_dim))
        self.bias_o = Parameter(torch.Tensor(hidden_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def _compute_cell(self, x, tu, su, t_up, t_low, t_up_diff, t_low_diff, s_up, s_low, s_up_diff, s_low_diff,
                      inputs_len, prev_hidden_state, prev_cell, predict):

        if predict == 0:
            # torch.ones_like: returns a tensor filled with the scalar value 1, with the same size as input.
            inputs_length1 = Variable(torch.ones_like(inputs_len))  # .to(torch.device('cpu'))
            Ttd = (t_up * ((t_up_diff / 60).view(len(t_up), 1)) + t_low * ((t_low_diff / 60).view(len(t_up), 1)))
            Std = (s_up * ((s_up_diff / 100).view(len(t_up), 1)) + s_low * ((s_low_diff / 100).view(len(t_up), 1)))

            x = x.float()

            Ttd = Ttd.float()
            Std = Std.float()

            i = (torch.sigmoid((x @ self.kernel_Wi) + (prev_hidden_state @ self.kernel_Ui) + (Ttd @ self.kernel_Ti) + (
                        Std @ self.kernel_Si) + self.bias_i)) * inputs_len  # ((tu)/168.0+(su)/34.0)*
            f = (torch.sigmoid((x @ self.kernel_Wf) + (prev_hidden_state @ self.kernel_Uf) + (Ttd @ self.kernel_Tf) + (
                        Std @ self.kernel_Sf) + self.bias_f)) * inputs_len
            o = (torch.sigmoid(
                (x @ self.kernel_Wog) + (prev_hidden_state @ self.kernel_Uog) + (Ttd @ self.kernel_To) + (
                            Std @ self.kernel_So) + self.bias_o)) * inputs_len

            C = (torch.tanh((x @ self.kernel_Wc) + (prev_hidden_state @ self.kernel_Uc) + self.bias_c)) * inputs_len

            # * prev_cell
            Ct = (f * prev_cell + i * C) * inputs_len + prev_cell * (inputs_length1 - inputs_len)
            current_hidden_state = (o * torch.tanh(Ct)) * inputs_len + prev_hidden_state * (inputs_length1 - inputs_len)
        else:
            x = x.float()
            i = torch.sigmoid((x @ self.kernel_Wi) + (prev_hidden_state @ self.kernel_Ui) + self.bias_i)  # i_t
            f = torch.sigmoid((x @ self.kernel_Wf) + (prev_hidden_state @ self.kernel_Uf) + self.bias_f)  # f_t
            o = torch.sigmoid((x @ self.kernel_Wog) + (prev_hidden_state @ self.kernel_Uog) + self.bias_o)  # o_t

            C = torch.tanh((x @ self.kernel_Wc) + (prev_hidden_state @ self.kernel_Uc) + self.bias_c)  # g_t

            # * prev_cell
            Ct = f * prev_cell + i * C
            current_hidden_state = o * torch.tanh(Ct)

        return current_hidden_state, Ct

    def forward(self, inputs1, tu, su, t_up, t_low, t_up_diff, t_low_diff, s_up, s_low, s_up_diff, s_low_diff,
                inputs_length, h, c, predict):  # prev_hidden_state, prev_cell):#state=None):

        current_hidden_state, Ct = self._compute_cell(inputs1, tu, su, t_up, t_low, t_up_diff, t_low_diff, s_up, s_low,
                                                      s_up_diff, s_low_diff, inputs_length, h, c, predict)

        return current_hidden_state, Ct  # cell_output
