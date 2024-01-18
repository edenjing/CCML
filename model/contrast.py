#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
import math
import torch.nn.functional as F


class UserEmbedding(nn.Module):
    def __init__(self):
        super(UserEmbedding, self).__init__()
        self.fc1 = torch.nn.Linear(40*40+1, 50)
        self.fc2 = torch.nn.Linear(50, 10)
        # self.sigmoid = F.sigmoid()

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        # emb_d1 = self.embedding_d1(user_wp)
        # ac_emb_d1 = F.sigmoid(emb_d1)
        # emb_d2 = self.embedding_d2(ac_emb_d1)
        return x


class UserContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(UserContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_user_wp, positive_user_wps, negative_user_wps):
        if positive_user_wps.size(0) == 0 or negative_user_wps.size(0) == 0:
            return torch.tensor(0.0)

        # positive_dis = torch.pairwise_distance(anchor_user_wp.unsqueeze(0), positive_user_wps.unsqueeze(0))
        # negative_dis = torch.pairwise_distance(anchor_user_wp.unsqueeze(0), negative_user_wps.unsqueeze(0))

        positive_dis = torch.zeros(positive_user_wps.size(0))
        negative_dis = torch.zeros(negative_user_wps.size(0))

        for i in range(positive_user_wps.size(0)):
            positive_dis[i] = torch.pairwise_distance(anchor_user_wp, positive_user_wps[i])

        for j in range(negative_user_wps.size(0)):
            negative_dis = torch.pairwise_distance(anchor_user_wp, negative_user_wps[j])

        loss = 0.5 * torch.pow(positive_dis.unsqueeze(1), 2) + \
               0.5 * torch.pow(torch.clamp(self.margin - negative_dis.unsqueeze(0), min=0.0), 2)

        return loss.mean()


# User Level Contrastive Learning
class UserClusteringContrast(nn.Module):
    def __init__(self, num_clusters, margin):
        super(UserClusteringContrast, self).__init__()
        # self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.margin = margin

        # User spatial distribution embedding layers for feature extraction
        self.UserEmbeddingLayers = UserEmbedding()
        # K-means Clustering
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        # Contrastive Loss
        self.contrastive_loss = UserContrastiveLoss(margin)

    def forward(self, anchor_user_wp, batch_user_wps, anchor_user_wp_id):
        # Feature extraction with fully connected layers
        anchor_embedding = self.UserEmbeddingLayers(anchor_user_wp).cpu()
        # batch_embeddings = self.UserEmbeddingLayers(batch_user_wps)

        batch_embeddings = torch.stack([self.UserEmbeddingLayers(user_wp) for user_wp in batch_user_wps])
        batch_embeddings_np = batch_embeddings.detach().cpu().numpy()

        # K-means clustering: iteratively assign each data point to the nearest centroid and update the centroids
        # based on the assigned points' mean value. After convergence, the final cluster centroids are obtained
        cluster_labels = self.kmeans.fit_predict(batch_embeddings_np)
        # Cluster label that containing anchor user embedding
        positive_cluster_label = cluster_labels[anchor_user_wp_id]
        # print("positive_cluster_label: ", positive_cluster_label)

        # Get positive and negative samples
        positive_embeddings = []
        for i, label in enumerate(cluster_labels):
            if i != anchor_user_wp_id and label == positive_cluster_label:
                positive_embeddings.append(batch_embeddings_np[i])
        negative_embeddings = batch_embeddings_np[cluster_labels != positive_cluster_label]

        positive_embeddings = torch.from_numpy(np.array(positive_embeddings))

        if positive_embeddings.size() == torch.Size([0]):
            positive_embeddings = positive_embeddings.unsqueeze(1).expand(-1, 10)

        negative_embeddings = torch.from_numpy(negative_embeddings)

        # Compute contrastive loss
        loss = self.contrastive_loss(anchor_embedding, positive_embeddings, negative_embeddings)

        return loss


class TrajContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(TrajContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.similarity_metric = nn.CosineSimilarity(dim=1)

    def forward(self, target_traj, augmented_trajs, negative_trajs):
        positive_pairs = self.create_positive_pairs(target_traj, augmented_trajs)
        negative_pairs = self.create_negative_pairs(target_traj, negative_trajs)

        positive_scores = self.compute_similarity_scores(positive_pairs)
        negative_scores = self.compute_similarity_scores(negative_pairs)

        # labels = torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)])
        # scores = torch.cat([positive_scores, negative_scores])

        loss = self.compute_loss(positive_scores, negative_scores)
        return loss

    def create_positive_pairs(self, target_traj, augmented_trajs):
        positive_pairs = []
        for a in target_traj:
            for b in augmented_trajs:
                 positive_pairs.append((a, b.squeeze()))
        return positive_pairs

    def create_negative_pairs(self, target_traj, negative_trajs):
        negative_pairs = []
        for a in target_traj:
            for b in negative_trajs:
                if b is not a:
                    negative_pairs.append((a, b.squeeze()))
        return negative_pairs

    def compute_similarity_scores(self, pairs):
        scores = []
        for pair in pairs:
            a, b = pair
            score = self.similarity_metric(a.unsqueeze(0), b.unsqueeze(0))
            scores.append(score)
        return torch.stack(scores)

    def compute_loss(self, positive_scores, negative_scores):

        positive_scores = positive_scores / self.temperature
        negative_scores = negative_scores / self.temperature

        # positive_log_probs = F.log_softmax(positive_scores, dim=1)
        # negative_log_probs = F.log_softmax(negative_scores, dim=1)

        # Reshape tensors to align dimensions
        # positive_log_probs = positive_log_probs.squeeze(1)  # Remove the dimension of size 1
        # negative_log_probs = negative_log_probs.squeeze(1)  # Remove the dimension of size 1

        # loss = -torch.mean(torch.diagonal(positive_log_probs) - torch.logsumexp(negative_log_probs, dim=0))

        loss = -torch.mean(torch.log(torch.exp(positive_scores) / torch.sum(torch.exp(negative_scores), dim=1)))
        return loss







