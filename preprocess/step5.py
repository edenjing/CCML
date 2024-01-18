#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
Step 5: generate the location/time_slot/dis_slot embedding files.
Original Design:
1. as all the trajectories are generated from one city road network, different users/tasks use shared embedding files.
2. we follow the framework of CHAML to use fixed embeddings during training.

Fix: In CHAML, different city has different POIs. For convenience, POI embedding is prepared in advance for each city.
However, in our work, there is no need to prepare the grid embeddings in advance.
'''

import os
import torch
import torch.nn as nn
import time
import numpy as np
from constants import *


class NullEmbedder(nn.Module):
    def __init__(self, loc_size, embed_dim):

        super(NullEmbedder, self).__init__()
        self.loc_embedding = nn.Embedding(loc_size, embed_dim)  # location information
        self.tslot_embedding = nn.Embedding(num_embeddings=92, embedding_dim=12)  # time interval information
        self.sslot_embedding = nn.Embedding(num_embeddings=92, embedding_dim=12)  # distance interval information

        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)  # Fills the input Tensor with a (semi) orthogonal matrix
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc_save_path, tslot_save_path, sslot_save_path):
        loc_w = self.loc_embedding.weight.detach().numpy()
        tslot_w = self.tslot_embedding.weight.detach().numpy()
        sslot_w = self.sslot_embedding.weight.detach().numpy()

        np.save(loc_save_path, loc_w)
        np.save(tslot_save_path, tslot_w)
        np.save(sslot_save_path, sslot_w)


def get_null_id_emb():
    EMBED_DIM = 100
    loc_size = 200 * 200 + 1
    null_embedder = NullEmbedder(loc_size, EMBED_DIM)
    embed_path = final_path + 'embeds/'

    if not os.path.exists(embed_path):
        os.mkdir(embed_path)

    loc_save_path = embed_path + "loc_embed.npy"
    tslot_save_path = embed_path + 'tslot_embed.npy'
    sslot_save_path = embed_path + 'sslot_embed.npy'

    null_embedder.forward(loc_save_path, tslot_save_path, sslot_save_path)


if __name__ == '__main__':
    start = time.time()
    get_null_id_emb()
    end = time.time()
    print("Step 5 finished, cost time:", (end - start) / 60, "min")