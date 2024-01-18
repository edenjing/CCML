#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import random
import torch


# Return: [traj, [next truth, des truth], len[pad with 1], length,
# [time_up, time_low, time_up_diff, time_low_diff], [dis_up, dis_low, dis_up_diff, dis_low_diff]]
def get_all_information_by_traj_idx(traj_all_information, traj_idx):
    all_information_by_traj_idx = []
    for i in range(len(traj_all_information)):
        if i == 1:
            all_information_by_traj_idx_i = []
            all_information_by_traj_idx_i.append(int(traj_all_information[i][0][traj_idx]))
            all_information_by_traj_idx_i.append(traj_all_information[i][1][traj_idx])

            all_information_by_traj_idx.append(all_information_by_traj_idx_i)
        elif (i == 4 or i == 5):
            all_information_by_traj_idx_i = []

            all_information_by_traj_idx_i.append(traj_all_information[i][0][traj_idx])
            all_information_by_traj_idx_i.append(traj_all_information[i][1][traj_idx])
            all_information_by_traj_idx_i.append(traj_all_information[i][2][traj_idx])
            all_information_by_traj_idx_i.append(traj_all_information[i][3][traj_idx])

            all_information_by_traj_idx.append(all_information_by_traj_idx_i)
        else:
            all_information_by_traj_idx.append(traj_all_information[i][traj_idx])
    return all_information_by_traj_idx


# get user ids of tasks/users
def fetch_user_ids(tasks, index=-1):
    users = tasks.keys()
    return users


# Input: get the spatial distributions for all generated users in each iteration
# user_ids_batch: generated user ids in each iteration
# users_wp: spatial distribution of all users
def fetch_batch_user_wps(target_user_id, batch_user_ids, users_wp,
                            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # Judge whether the user distribution exists
    if target_user_id in users_wp.keys():
        target_user_wp = users_wp[target_user_id]
    else:
        target_user_wp = [0 for _ in range(40 * 40 + 1)]
    target_user_wp = torch.Tensor(target_user_wp).to(device)

    batch_user_wps = []
    for i, user_id in enumerate(batch_user_ids):
        if user_id == target_user_id:
            target_user_id = i
        if user_id in users_wp.keys():
            user_id_wp = users_wp[user_id]
        else:
            user_id_wp = [0 for _ in range(40 * 40 + 1)]
        user_id_wp = torch.Tensor(user_id_wp).to(device)
        batch_user_wps.append(user_id_wp)

    return target_user_wp, batch_user_wps, target_user_id


def fetch_sameo_data(sameo_tasks_data, task_ids):
    new_sameo_tasks_data = []
    for task_id in task_ids:
        new_sameo_tasks_data.append(sameo_tasks_data[task_id])
    return new_sameo_tasks_data


# get corresponding embeddings of selected tasks/trajs
# def fetch_embeds(task_ids, task_sample_id2traj_id, embeds):
#     # embeds = [loc_emb, tslot_emb, sslot_emb]
#     for task in task_ids:
#     return selected_embeds

# Return: all information of generated tasks (ordered by user id)
# For each user: [0: spt all info, 1: qry all info]

# [spt all info]: order by traj id: [traj, [next truth, des truth], len[pad with 1], length,
# [time_up, time_low, time_up_diff, time_low_diff], [dis_up, dis_low, dis_up_diff, dis_low_diff]]

# [qry all info]: order by traj id: [traj, [next truth, des truth], len[pad with 1], length,
# [time_up, time_low, time_up_diff, time_low_diff], [dis_up, dis_low, dis_up_diff, dis_low_diff]]
class TrainGenerator():
    def __init__(self, mtrain_tasks, task_batch_size, curriculum_user_ids,
                 pacing_function='ssp', few_traj_num=200, max_steps=None):
        self.users = fetch_user_ids(mtrain_tasks, index=-1)
        self.task_num = len(self.users)
        self.task_batch_size = task_batch_size  # size of tasks (users) in a batch
        self.mtrain_tasks = mtrain_tasks
        self.curriculum_user_ids = curriculum_user_ids
        self.max_steps = max_steps
        self.pacing_function = pacing_function
        self.few_traj_num = few_traj_num   # the number of selected trajs for each user/task

    def fetch_task_batch(self, task_id2results=None, stage="stage1", curriculum=True, hard_task=True, train_step=0):
        task_id2acc = task_id2results["task_id2acc"]
        task_id_to_traj2acc = task_id2results["task_id_to_traj2acc"]
        '''
        Single Step Pacing:
        1. First sort the base users from easiest to hardest by the criteria of user-level difficulty;
        2. Then for the first A out of M iterations, we only present B easiest base users as user sampling pool;
        3. Lastly, for the rest of iterations, we present all the base users.
        '''
        # 1. User-level curriculum decides the sampling pool of meta-training tasks.
        if curriculum:
            if self.pacing_function == 'ssp':
                starting_percent = 0.5
                step_length = self.max_steps // 2  # for what?
                if train_step < step_length:
                    gi = int(starting_percent * self.task_num)
                else:
                    gi = self.task_num
                # task index -> user id
                # curriculum_user_ids should be a list of strings
                task_ids_pool = self.curriculum_user_ids[:gi]
        else:
            task_ids_pool = list(range(self.task_num))

        self.task_ids_pool = task_ids_pool
        # self.last_batch_id = train_step

        # 2. Decide to sample which users/tasks for this round.
        # None means the it's the first iteration of meta-training,
        # then we randomly sample tasks from the current user pool
        if task_id2acc is None:
            task_ids = random.sample(task_ids_pool, k=self.task_batch_size)
        # for Stage 2, we keep the same group of users (tasks), and sample new trajs
        elif stage == "stage2":
            task_ids = list(task_id2acc.keys())
        # for the rest of meta-training iterations, if we adopt hard_task strategy, we try to sample harder users
        # 保留上一轮Stage 2中的一半困难的tasks，再随机sample一些其他的tasks
        elif hard_task:
            hard_task_num = self.task_batch_size // 2
            task_ids = list(task_id2acc.keys())[:hard_task_num]  # the hardest tasks of last training user batch
            # 如果user pool里的所有users减去上一轮Stage2中的user的数量 < 需要再few_user_num随机sample的user数量，
            # 就在current user pool里面直接sample task_batch_size个new users
            # 否则就从current user pool里减去上一轮Stage2中的users，在剩下的users里面随机sample需要的new users
            if len(task_ids_pool) - self.task_batch_size < self.task_batch_size - hard_task_num:
                task_ids = random.sample(task_ids_pool, k=self.task_batch_size)
            else:
                other_task_pool = list(set(task_ids_pool)-set(list(task_id2acc.keys())))
                other_task_ids = random.sample(other_task_pool, k=self.task_batch_size - hard_task_num)
                task_ids.extend(other_task_ids)
        else:
            task_ids = random.sample(task_ids_pool, k=self.task_batch_size)

        # print("len(task_ids): ", len(task_ids), "task_ids: ", task_ids)

        # 3. Decide the trajs of each user (to form support&query set samples).
        # each task/user -> a list x, x[i] means the traj id of the i-th sample
        task_sample_id2traj_id = []

        # order by user
        generated_task_batch = {}

        # task_idxs stores user ids
        for task_id in task_ids:
            # print("get information of train_task[{}]...".format(task_id))
            # mtrain_task[id]: userid_spt_qry_all_information = [spt_all_information, qry_all_information]
            spt_traj_all_information = self.mtrain_tasks[task_id][0]
            qry_traj_all_information = self.mtrain_tasks[task_id][1]
            # get the traj idxs of current user/task
            # spt_traj_all_information[0] is the trajectory numbers of current user/task
            all_traj_idxs = list(np.arange(len(spt_traj_all_information[0])))
            # print("all_traj_idxs: ", all_traj_idxs)

            if stage == "stage2" and (task_id_to_traj2acc is not None) and (task_id in task_id_to_traj2acc):
                traj2acc = task_id_to_traj2acc[task_id]
                # keep a half of difficult trajs in stage 2 of last iteration
                # $k_j$ is 0.5, $B_j$ is self.few_traj_num
                hard_traj_num = self.few_traj_num // 2
                selected_trajs = list(traj2acc.keys())[:hard_traj_num]
                other_traj_pool = list(set(all_traj_idxs) - set(list(traj2acc.keys())))
                other_trajs = random.sample(other_traj_pool, k=self.few_traj_num - hard_traj_num)
                selected_trajs.extend(other_trajs)
            else:
                selected_trajs = random.sample(all_traj_idxs, k=self.few_traj_num)

            # print("idxs of selected_trajs: ", selected_trajs)

            # construct the support set samples of a meta-training task
            # which actually means extract all information of selected_trajs

            # order by traj indexes in selected_trajs
            spt_sample_all_information = []
            for traj_idx in selected_trajs:
                # get all information for one traj
                traj_spt_all_information = get_all_information_by_traj_idx(spt_traj_all_information, traj_idx)
                spt_sample_all_information.append(traj_spt_all_information)
            # print("spt_sample_all_information construction finished.")

            # here may insert the negative and positive samples
            # samples2_pos_and_neg(loc_spt_samples, tslot_spt_samples, sslot_spt_samples)
            # samples_to_input(loc_spt_samples, tslot_spt_samples, sslot_spt_samples)

            # construct the query set samples of a meta-training task
            qry_sample_all_information = []
            for traj_idx in selected_trajs:
                # get all information for one traj
                traj_qry_all_information = get_all_information_by_traj_idx(qry_traj_all_information, traj_idx)
                qry_sample_all_information.append(traj_qry_all_information)
            # print("qry_sample_all_information construction finished.")

            task_sample_id2traj_id.append(selected_trajs)

            # append the (spt, qry) information (ordered by traj id) for current task
            generated_task_batch[task_id] = [spt_sample_all_information, qry_sample_all_information]
            # print("append one new task end.")
        # print("task_sample_id2traj_id: ", task_sample_id2traj_id)
        # print("Training tasks data generation finished.")
        return generated_task_batch, task_ids, task_sample_id2traj_id









