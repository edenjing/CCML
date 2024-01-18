#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
try:
    from learner import Learner
except:
    from model.learner import Learner
from utils.utils import *
from copy import deepcopy
from time import sleep
from tqdm import tqdm
from torch.autograd import Variable
from utils.metrics import Metrics
from utils.metadataset import fetch_user_ids, get_all_information_by_traj_idx, fetch_batch_user_wps, fetch_sameo_data
from model.contrast import UserClusteringContrast, TrajContrastiveLoss
from utils.dataaugment import generate_augmented_traj, combine_all_augmented_info


class Meta(nn.Module):
    '''
    CMCTP Meta Learner.
    '''
    def __init__(self, config):
        super(Meta, self).__init__()
        self.update_lr = config['update_lr']  # task-level inner update learning rate
        self.meta_lr = config['meta_lr']  # meta-level outer learning rate
        self.update_step = config['update_step']  # task-level inner update steps
        self.update_step_test = config['update_step_test']  # update steps for fine tuning
        # vars[0:LOCAL_FIX_VAR] should be fixed in the local update (fast_weights only update [LOCAL_FIX_VAR:])
        # self.LOCAL_FIX_VAR = config['local_fix_var']
        # self.sample_batch_size = config['sample_batch_size']  # batch size of samples to feed into Learner
        self.top_k = config['top_k']
        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.USE_CUDA else "cpu")
        self.BASEModel = Learner(config, self.device)
        self.traj_level_loss = torch.nn.NLLLoss(ignore_index=0)  # The negative log likelihood loss.
        self.loss_coef = config['loss_coef']
        self.UserClusteringContrast = UserClusteringContrast(config['num_clusters'], config['user_contrast_margin'])
        self.TrajContrastiveLoss = TrajContrastiveLoss(config['temperature'])

        self.mask_bool = config['mask_bool']
        self.truncate_bool = config['truncate_bool']
        self.two_hop_bool = config['two_hop_bool']
        self.detour_bool = config['detour_bool']

        self.mask_ratio = config['mask_ratio']
        self.truncate_ratio = config['truncate_ratio']
        self.replace_ratio = config['replace_ratio']

    # def store_parameters(self):
    #     self.keep_weight = deepcopy(self.model.state_dict())
    #     self.weight_name = list(self.keep_weight.keys())
    #     self.weight_len = len(self.keep_weight)
    #     self.fast_weights = OrderedDict()

    # prepare for model input format
    # all_trajs_info: [[target traj], [augmented trajs], [negative trajs]]
    # Output: Dataloaders[[target traj dataloaders], [augmented trajs dataloaders], [negative trajs dataloaders]]
    # dataloaders: [traj DATA_LOADER]
    def data_operation(self, all_trajs_info):
        processed_data = []  # target traj, augmented trajs, negative trajs
        augment_num = len(all_trajs_info[1])
        negative_num = len(all_trajs_info[2])

        Dataloaders = []
        for type_item in all_trajs_info:
            dataloaders = []
            for traj in type_item:
                traj_input = data_to_device(traj)

                traj, next_truth, des_truth, len_pad, length, \
                time_up, time_low, time_up_diff, time_low_diff, \
                dis_up, dis_low, dis_up_diff, dis_low_diff = traj_input

                # Perform TrajDataLoader
                DATA = TrajDataLoader(traj, len_pad, length, time_up, time_low, time_up_diff, time_low_diff,
                                          dis_up, dis_low, dis_up_diff, dis_low_diff, next_truth, des_truth)

                DATA_LOADER = DataLoader(DATA, batch_size=1)

                dataloaders.append(DATA_LOADER)
            Dataloaders.append(dataloaders)
        return Dataloaders, augment_num, negative_num

    # data[task_id] = [spt_sample_all_information, qry_sample_all_information]
    # task_ids -> a list A, stores user ids in a task batch
    # task_sample_id2traj_id -> a 2D list B, each B[i] stores the selected traj indexes for user(task) A[i]
    def forward(self, data, sameo_tasks_data, task_ids, users_wp, optimizer, parameters):
        task_num = len(data)
        print("train task number: ", task_num)
        # record the prediction accuracy on task level
        # count一个task里面有多少条traj的predicted location排在tok-k内，越说明这个task容易学习
        task_level_acc = []
        task_traj_level_acc = []
        task_level_losses = []

        optimizer.zero_grad()

        for task_i in range(task_num):
            # torch.save(self.BASEModel.state_dict(), './data/model_save/model_params.pth')
            # self.BASEModel.load_state_dict(torch.load('./data/model_save/model_params.pth'))
            # print("task_i (user_i): ", task_i)
            task_id = task_ids[task_i]
            # Run the i-th task (which contains different number of trajs)
            spt_all_info = data[task_id][0]  # order by traj index
            qry_all_info = data[task_id][1]

            # get spatial distributions for current user and the remaining users in current iteration
            # batch_users_wps contains cur_user_wp
            # cur_user_wp_id = task_id
            cur_user_wp, batch_users_wps, cur_user_wp_id = fetch_batch_user_wps(task_id, task_ids, users_wp, self.device)
            user_level_loss = self.UserClusteringContrast(cur_user_wp, batch_users_wps, cur_user_wp_id)

            task_id_sameo_data = sameo_tasks_data[task_i]
            # traj_level_losses should be the final mean loss of all trajs in current task (user)
            traj_level_losses, traj_level_acc = self.localupdate(spt_all_info, qry_all_info, optimizer, task_id_sameo_data)
            # print("traj_level_losses:", traj_level_losses)
            # grad = torch.autograd.grad(traj_level_losses, self.BASEModel.parameters())
            # self.BASEModel.update_parameters(grad)

            # fast_weights = list(self.BASEModel.parameters())[:self.LOCAL_FIX_VAR] \
            #                + list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad[self.LOCAL_FIX_VAR:], self.BASEModel.parameters()[self.LOCAL_FIX_VAR:])))
            task_loss = torch.mean(torch.stack(traj_level_losses))
            # add the user-level contrastive loss
            task_loss = task_loss + user_level_loss

            task_loss = task_loss.clone().detach().requires_grad_(True)
            if torch.isnan(task_loss).any():
                print("nan exists in tensor (task_loss in forward).")
            print("task_loss: {}".format(task_i), task_loss.item())

            task_level_losses.append(task_loss)
            # traj level accuracy
            task_traj_level_acc.append(traj_level_acc)
            # task level accuracy
            top_k_count = traj_level_acc.count(rk for rk in traj_level_acc if rk <= self.top_k)
            # print("top_k_count: ", top_k_count)
            task_level_acc.append(top_k_count)

        # total_loss should be the final mean loss of all tasks
        total_loss = torch.mean(torch.stack(task_level_losses))

        if torch.isnan(total_loss).any():
                print("nan exists in tensor (total_loss in forward).")

        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)

        torch.autograd.set_detect_anomaly(True)

        torch.nn.utils.clip_grad_value_(parameters, 0.25)
        optimizer.step()

        acc_results = {"task_level_acc": task_level_acc,
                       "task_traj_level_acc": task_traj_level_acc,}
        # return total_loss * (1 / self.loss_coef), acc_results
        return total_loss, acc_results

    '''
    Originally, the prediction of one trajectory is one task, the gradient is computed and the network is updated accordingly.
    Now, we take the prediction of one user as one task and add a level of sub-tasks. we sum up the loss of all sub-tasks and compute their average value. 
    Then we use this mean value as the final loss of this user-level task to perform the gradient computation and network parameter update. 
    '''

    # local update for one task, which contains amount of trajs
    def localupdate(self, spt_set_info, qry_set_info, optimizer, task_id_sameo_data):
        traj_level_losses = []
        # record the prediction accuracy on traj level
        # 对于一条traj来说，ground truth location在预测的结果中的index（即rank），越靠前越说明这条traj is easier to learn
        traj_level_acc = []

        # set progress bar
        # progress_bar = tqdm(total=len(spt_set_info), desc="Training", unit="Traj")

        progress_bar = tqdm(total=len(spt_set_info), desc="Training", unit="Traj", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}', ncols=80)

        for up_step in range(self.update_step):
            # task_level_loss should be the mean value of all traj tasks
            # user_level_loss = []

            for i in range(len(spt_set_info)):
                # spt
                spt_all_trajs_info = []  # reordered: [original traj, augmented trajs, negative trajs]
                spt_i_info = spt_set_info[i]  # original traj info
                spt_negative_trajs_info = spt_set_info[:i] + spt_set_info[i+1:]  # negative trajs info

                spt_masked_info, spt_truncated_info, spt_two_hopped_info, spt_detoured_info = generate_augmented_traj(
                    spt_i_info, mask_bool=self.mask_bool, truncate_bool=self.truncate_bool, two_hop_bool=self.two_hop_bool, detour_bool=self.detour_bool,
                    mask_ratio=self.mask_ratio, truncate_ratio=self.truncate_ratio, replace_ratio=self.replace_ratio, traj_type='spt', user_id_data=task_id_sameo_data)

                # combine original traj and augmented traj
                # order: original info, masked info, truncated_info, two_hopped_info, detoured_info
                spt_combined_augmented_info = combine_all_augmented_info(spt_i_info, spt_masked_info, spt_truncated_info, spt_two_hopped_info, spt_detoured_info)

                spt_all_trajs_info.append([spt_i_info])
                spt_all_trajs_info.append(spt_combined_augmented_info)
                spt_all_trajs_info.append(spt_negative_trajs_info)

                # qry
                qry_all_trajs_info = []
                qry_i_info = qry_set_info[i]
                qry_negative_trajs_info = qry_set_info[:i] + qry_set_info[i+1:]

                qry_masked_info, qry_truncated_info, qry_two_hopped_info, qry_detoured_info = generate_augmented_traj(
                    qry_i_info, mask_bool=self.mask_bool, truncate_bool=self.truncate_bool, two_hop_bool=self.two_hop_bool, detour_bool=self.detour_bool,
                    mask_ratio=self.mask_ratio, truncate_ratio=self.truncate_ratio, replace_ratio=self.replace_ratio, traj_type='qry', user_id_data=task_id_sameo_data)

                # combine original traj and augmented traj
                # order: original info, masked info, truncated_info, two_hopped_info, detoured_info
                qry_combined_augmented_info = combine_all_augmented_info(qry_i_info, qry_masked_info, qry_truncated_info, qry_two_hopped_info, qry_detoured_info)

                qry_all_trajs_info.append([qry_i_info])
                qry_all_trajs_info.append(qry_combined_augmented_info)
                qry_all_trajs_info.append(qry_negative_trajs_info)

                # Output: Dataloaders[[target traj dataloaders], [augmented trajs dataloaders], [negative trajs dataloaders]]
                # dataloaders: [traj DATA_LOADER]
                spt_Dataloaders, spt_augment_num, spt_negative_num = self.data_operation(spt_all_trajs_info)
                qry_Dataloaders, qry_augment_num, qry_negative_num = self.data_operation(qry_all_trajs_info)

                # local update
                spt_Pres = []
                spt_Reps = []
                for dataloaders in spt_Dataloaders:
                    Pres = []
                    Reps = []
                    for spt_DATA_LOADER in dataloaders:

                        optimizer.zero_grad()

                        for spt_batch_i, (spt_traj_i, spt_len_pad_i, spt_length_i, spt_time_up_i, spt_time_low_i, spt_time_up_diff_i, spt_time_low_diff_i,
                                          spt_dis_up_i, spt_dis_low_i, spt_dis_up_diff_i, spt_dis_low_diff_i, spt_next_truth_i, spt_des_truth_i) in enumerate(spt_DATA_LOADER):

                            spt_traj_i, spt_next_truth_i, spt_des_truth_i, spt_len_pad_i, spt_length_i, \
                            spt_time_up_i, spt_time_low_i, spt_time_up_diff_i, spt_time_low_diff_i, \
                            spt_dis_up_i, spt_dis_low_i, spt_dis_up_diff_i, spt_dis_low_diff_i = \
                                Variable(spt_traj_i).to(self.device), Variable(spt_next_truth_i).to(self.device), Variable(spt_des_truth_i).to(self.device), \
                                Variable(spt_len_pad_i).to(self.device), Variable(spt_length_i).to(self.device), Variable(spt_time_up_i).to(self.device), \
                                Variable(spt_time_low_i).to(self.device), Variable(spt_time_up_diff_i).to(self.device), Variable(spt_time_low_diff_i).to(self.device), \
                                Variable(spt_dis_up_i).to(self.device), Variable(spt_dis_low_i).to(self.device), Variable(spt_dis_up_diff_i).to(self.device), \
                                Variable(spt_dis_low_diff_i).to(self.device)
                            # local update
                            # print("local update for traj {}...".format(traj_i))
                            pred_y_next, traj_rep = self.BASEModel(spt_traj_i, spt_len_pad_i, spt_length_i, spt_time_up_i, spt_time_low_i, spt_time_up_diff_i,
                                                             spt_time_low_diff_i, spt_dis_up_i, spt_dis_low_i, spt_dis_up_diff_i, spt_dis_low_diff_i)
                            # print("pred_y_next: ", pred_y_next)
                            # print("spt_next_truth: ", spt_next_truth_i)

                            Pres.append(pred_y_next)
                            Reps.append(traj_rep)

                    spt_Pres.append(Pres)
                    spt_Reps.append(Reps)

                traj_predict_loss = self.traj_level_loss(spt_Pres[0][0], spt_next_truth_i.long())
                traj_contrast_loss = self.TrajContrastiveLoss(spt_Reps[0][0], spt_Reps[1], spt_Reps[2])

                # print("traj_predict_loss: ", traj_predict_loss)
                # print("traj_contrast_loss: ", traj_contrast_loss)

                traj_loss = traj_predict_loss + traj_contrast_loss

                if torch.isnan(traj_loss).any():
                    print("nan exists in tensor (support traj in local update).")

                # check the correctness
                # self.BASEModel.zero_grad()
                traj_loss.backward(retain_graph=True)

                # print("traj_loss (support set in local update): ", traj_loss)

                with torch.no_grad():
                    for param in self.BASEModel.parameters():
                        param_copy = param.clone()
                        param_copy = param_copy - self.update_lr * param.grad
                        param.data.copy_(param_copy)

                        # for param, param_copy in zip(self.BASEModel.parameters(), params_copy):
                        #     param.copy_(param_copy)
                        #     param -= self.update_lr * param.grad

                optimizer.step()

                qry_Pres = []
                qry_Reps = []
                for dataloaders in qry_Dataloaders:
                    Pres = []
                    Reps = []
                    for qry_DATA_LOADER in dataloaders:

                        for qry_batch_i, (qry_traj_i, qry_len_pad_i, qry_length_i, qry_time_up_i, qry_time_low_i, qry_time_up_diff_i, qry_time_low_diff_i,
                                          qry_dis_up_i, qry_dis_low_i, qry_dis_up_diff_i, qry_dis_low_diff_i, qry_next_truth_i, qry_des_truth_i) in enumerate(qry_DATA_LOADER):

                            qry_traj_i, qry_next_truth_i, qry_des_truth_i, qry_len_pad_i, qry_length_i, \
                            qry_time_up_i, qry_time_low_i, qry_time_up_diff_i, qry_time_low_diff_i, \
                            qry_dis_up_i, qry_dis_low_i, qry_dis_up_diff_i, qry_dis_low_diff_i = \
                                Variable(qry_traj_i).to(self.device), Variable(qry_next_truth_i).to(self.device), Variable(qry_des_truth_i).to(self.device), \
                                Variable(qry_len_pad_i).to(self.device), Variable(qry_length_i).to(self.device), Variable(qry_time_up_i).to(self.device), \
                                Variable(qry_time_low_i).to(self.device), Variable(qry_time_up_diff_i).to(self.device), Variable(qry_time_low_diff_i).to(self.device), \
                                Variable(qry_dis_up_i).to(self.device), Variable(qry_dis_low_i).to(self.device), Variable(qry_dis_up_diff_i).to(self.device), \
                                Variable(qry_dis_low_diff_i).to(self.device)

                            # local update
                            # print("local update for traj {}...".format(traj_i))
                            pred_y_next, traj_rep = self.BASEModel(qry_traj_i, qry_len_pad_i, qry_length_i,
                                                                   qry_time_up_i, qry_time_low_i, qry_time_up_diff_i,
                                                                   qry_time_low_diff_i, qry_dis_up_i, qry_dis_low_i,
                                                                   qry_dis_up_diff_i, qry_dis_low_diff_i)
                            # print("pred_y_next: ", pred_y_next)
                            # print("spt_next_truth: ", spt_next_truth_i)
                            Pres.append(pred_y_next)
                            Reps.append(traj_rep)
                    qry_Pres.append(Pres)
                    qry_Reps.append(Reps)

                traj_predict_loss = self.traj_level_loss(qry_Pres[0][0], qry_next_truth_i.long())
                traj_contrast_loss = self.TrajContrastiveLoss(qry_Reps[0][0], qry_Reps[1], qry_Reps[2])

                traj_loss = traj_predict_loss + traj_contrast_loss

                if torch.isnan(traj_loss).any():
                    print("nan exists in tensor (query traj in local update).")

                # get traj level acc
                # pred_probs = np.array(q_pred_y_next[0].tolist())
                pred_probs = np.array(qry_Pres[0][0][0].tolist())
                # get the index by item descending order
                descend_ranki = np.argsort(pred_probs)[::-1]
                # smaller pred_ranki, more accurate prediction
                # pred_ranki = np.where(descend_ranki == qry_next_truth_i.item())[0] + 1
                pred_ranki = descend_ranki.tolist().index(qry_next_truth_i.item()) + 1
                traj_level_acc.append(pred_ranki)

                # traj_level_losses.append(traj_loss*self.loss_coef)
                traj_level_losses.append(traj_loss)

                sleep(0.5)
                # update the progress bar
                progress_bar.update(1)
        progress_bar.close()

        return traj_level_losses, traj_level_acc

    # Here the task_dataset can be valid_task or test_tasks
    # As the number of valid tasks(84) and test tasks(100) are small, we validate and test on all data
    def evaluate(self, task_dataset, sameo_tasks_data, metric, users_wp, test=False, few_task_num=100, few_traj_num=100):
        # The weight of each task in the final evaluation scores,
        # according to the amount of query trajs in this task
        task_scores = []
        task_score_weights = []  # the number of trajectories in each task
        task_level_losses = []

        task_ids = fetch_user_ids(task_dataset)
        task_level_acc = []
        task_traj_level_acc = []

        meta_model1 = deepcopy(self.BASEModel)

        if len(task_ids) > few_task_num:
            task_ids = list(task_ids)[:few_task_num]

        # evaluate on one user (task) each time
        for task_i in range(len(task_ids)):
            task_id = task_ids[task_i]
            # print("evaluate task {}...".format(task_id))

            # construct the selected valid/test data into model input format: for each user, all info are ordered by order
            spt_sample_all_info = []
            qry_sample_all_info = []

            if(len(task_dataset[task_id][0][0]) < few_traj_num):
                few_traj_num = len(task_dataset[task_id][0][0])

            for traj_idx in range(few_traj_num):
                # get all information for one traj
                traj_spt_all_information = get_all_information_by_traj_idx(task_dataset[task_id][0], traj_idx)
                spt_sample_all_info.append(traj_spt_all_information)

                traj_qry_all_information = get_all_information_by_traj_idx(task_dataset[task_id][1], traj_idx)
                qry_sample_all_info.append(traj_qry_all_information)

            # The score is determined by the number of trajectories per task
            task_score_weights.append(few_traj_num)

            cur_user_wp, batch_users_wps, cur_user_wp_id = fetch_batch_user_wps(task_id, task_ids, users_wp, self.device)
            user_level_loss = self.UserClusteringContrast(cur_user_wp, batch_users_wps, cur_user_wp_id)

            task_id_sameo_data = fetch_sameo_data(sameo_tasks_data, task_ids)

            task_id_sameo_data = task_id_sameo_data[task_i]

            traj_level_losses, traj_level_acc = self.localvalid(spt_sample_all_info, qry_sample_all_info, meta_model1=meta_model1, task_id_sameo_data=task_id_sameo_data)
            task_loss = torch.mean(torch.stack(traj_level_losses))
            task_loss = task_loss + user_level_loss

            task_level_losses.append(task_loss)

            # print("valid task: ", task_id, "valid loss: ", task_loss.items())

            # score version 1
            # task_traj_level_acc.append(traj_level_acc)
            # task_scores.append(np.mean(traj_level_acc))
            # top_k_count = traj_level_acc.count(rk for rk in traj_level_acc if rk <= top_k)
            # task_level_acc.append(top_k_count)

            scores = metric.compute_scores(traj_level_acc)
            print("{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{}".format(*scores, task_id))
            task_scores.append(scores)

        # final loss on all tasks
        total_loss = torch.mean(torch.stack(task_level_losses))
        print("task_score_weights: ", task_score_weights)

        if test == False:
            return total_loss, task_scores, task_score_weights
        else:
            return task_scores, task_score_weights

    # local valid for one task, which can belong to mvalid_tasks or mtest_tasks
    def localvalid(self, spt_all_info, qry_all_info, meta_model1, task_id_sameo_data):
        traj_level_losses = []
        traj_level_acc = []

        # q_pred_y_next_list = []  # store all the prediction results (sorted prediction probabilities)
        for up_step in range(self.update_step_test):

            for i in range(len(spt_all_info)):
                # spt
                spt_all_trajs_info = []  # reordered: [original traj, augmented trajs, negative trajs]
                spt_i_info = spt_all_info[i]  # original traj info
                spt_negative_trajs_info = spt_all_info[:i] + spt_all_info[i+1:]  # negative trajs info

                spt_masked_info, spt_truncated_info, spt_two_hopped_info, spt_detoured_info = generate_augmented_traj(spt_i_info, mask_bool=self.mask_bool, truncate_bool=self.truncate_bool,
                    two_hop_bool=self.two_hop_bool, detour_bool=self.detour_bool, mask_ratio=self.mask_ratio, truncate_ratio=self.truncate_ratio, replace_ratio=self.replace_ratio,
                    traj_type='spt', user_id_data=task_id_sameo_data)

                spt_combined_augmented_info = combine_all_augmented_info(spt_i_info, spt_masked_info, spt_truncated_info, spt_two_hopped_info, spt_detoured_info)

                spt_all_trajs_info.append([spt_i_info])
                spt_all_trajs_info.append(spt_combined_augmented_info)
                spt_all_trajs_info.append(spt_negative_trajs_info)

                # qry
                qry_all_trajs_info = []
                qry_i_info = qry_all_info[i]
                qry_negative_trajs_info = qry_all_info[:i] + qry_all_info[i+1:]

                qry_masked_info, qry_truncated_info, qry_two_hopped_info, qry_detoured_info = generate_augmented_traj(qry_i_info, mask_bool=self.mask_bool, truncate_bool=self.truncate_bool,
                    two_hop_bool=self.two_hop_bool, detour_bool=self.detour_bool, mask_ratio=self.mask_ratio, truncate_ratio=self.truncate_ratio, replace_ratio=self.replace_ratio,
                    traj_type='qry', user_id_data=task_id_sameo_data)

                qry_combined_augmented_info = combine_all_augmented_info(qry_i_info, qry_masked_info, qry_truncated_info, qry_two_hopped_info, qry_detoured_info)

                qry_all_trajs_info.append([qry_i_info])
                qry_all_trajs_info.append(qry_combined_augmented_info)
                qry_all_trajs_info.append(qry_negative_trajs_info)

                spt_Dataloaders, spt_augment_num, spt_negative_num = self.data_operation(spt_all_trajs_info)
                qry_Dataloaders, qry_augment_num, qry_negative_num = self.data_operation(qry_all_trajs_info)

                # local update
                spt_Pres = []
                spt_Reps = []
                for dataloaders in spt_Dataloaders:
                    Pres = []
                    Reps = []
                    for spt_DATA_LOADER in dataloaders:
                        for spt_batch_i, (spt_traj_i, spt_len_pad_i, spt_length_i, spt_time_up_i, spt_time_low_i, spt_time_up_diff_i, spt_time_low_diff_i,
                                          spt_dis_up_i, spt_dis_low_i, spt_dis_up_diff_i, spt_dis_low_diff_i, spt_next_truth_i, spt_des_truth_i) in enumerate(spt_DATA_LOADER):

                            spt_traj_i, spt_next_truth_i, spt_des_truth_i, spt_len_pad_i, spt_length_i, \
                            spt_time_up_i, spt_time_low_i, spt_time_up_diff_i, spt_time_low_diff_i, \
                            spt_dis_up_i, spt_dis_low_i, spt_dis_up_diff_i, spt_dis_low_diff_i = \
                                Variable(spt_traj_i).to(self.device), Variable(spt_next_truth_i).to(self.device), Variable(spt_des_truth_i).to(self.device), \
                                Variable(spt_len_pad_i).to(self.device), Variable(spt_length_i).to(self.device), Variable(spt_time_up_i).to(self.device), \
                                Variable(spt_time_low_i).to(self.device), Variable(spt_time_up_diff_i).to(self.device), Variable(spt_time_low_diff_i).to(self.device), \
                                Variable(spt_dis_up_i).to(self.device), Variable(spt_dis_low_i).to(self.device), Variable(spt_dis_up_diff_i).to(self.device), \
                                Variable(spt_dis_low_diff_i).to(self.device)
                            # local update
                            # print("local update for traj {}...".format(traj_i))
                            pred_y_next, traj_rep = self.BASEModel(spt_traj_i, spt_len_pad_i, spt_length_i, spt_time_up_i, spt_time_low_i, spt_time_up_diff_i,
                                                             spt_time_low_diff_i, spt_dis_up_i, spt_dis_low_i, spt_dis_up_diff_i, spt_dis_low_diff_i)
                            # print("pred_y_next: ", pred_y_next)
                            # print("spt_next_truth: ", spt_next_truth_i)

                            Pres.append(pred_y_next)
                            Reps.append(traj_rep)

                    spt_Pres.append(Pres)
                    spt_Reps.append(Reps)

                traj_predict_loss = self.traj_level_loss(spt_Pres[0][0], spt_next_truth_i.long())
                traj_contrast_loss = self.TrajContrastiveLoss(spt_Reps[0][0], spt_Reps[1], spt_Reps[2])

                traj_loss = traj_predict_loss + traj_contrast_loss

                meta_model1.zero_grad()
                grad = torch.autograd.grad(traj_loss, meta_model1.parameters(),
                                           allow_unused=True)  # , create_graph=True

                with torch.no_grad():
                    i = 0
                    for param in meta_model1.parameters():
                        param_copy = param.clone()
                        # print("param_copy", param_copy)
                        # print("self.update_lr:", self.update_lr)
                        param_copy = param_copy - self.update_lr * grad[i]
                        param.data.copy_(param_copy)
                        i += 1

                qry_Pres = []
                qry_Reps = []
                for dataloaders in qry_Dataloaders:
                    Pres = []
                    Reps = []
                    for qry_DATA_LOADER in dataloaders:

                        for qry_batch_i, (qry_traj_i, qry_len_pad_i, qry_length_i, qry_time_up_i, qry_time_low_i, qry_time_up_diff_i, qry_time_low_diff_i,
                                          qry_dis_up_i, qry_dis_low_i, qry_dis_up_diff_i, qry_dis_low_diff_i, qry_next_truth_i, qry_des_truth_i) in enumerate(qry_DATA_LOADER):

                            qry_traj_i, qry_next_truth_i, qry_des_truth_i, qry_len_pad_i, qry_length_i, \
                            qry_time_up_i, qry_time_low_i, qry_time_up_diff_i, qry_time_low_diff_i, \
                            qry_dis_up_i, qry_dis_low_i, qry_dis_up_diff_i, qry_dis_low_diff_i = \
                                Variable(qry_traj_i).to(self.device), Variable(qry_next_truth_i).to(self.device), Variable(qry_des_truth_i).to(self.device), \
                                Variable(qry_len_pad_i).to(self.device), Variable(qry_length_i).to(self.device), Variable(qry_time_up_i).to(self.device), \
                                Variable(qry_time_low_i).to(self.device), Variable(qry_time_up_diff_i).to(self.device), Variable(qry_time_low_diff_i).to(self.device), \
                                Variable(qry_dis_up_i).to(self.device), Variable(qry_dis_low_i).to(self.device), Variable(qry_dis_up_diff_i).to(self.device), \
                                Variable(qry_dis_low_diff_i).to(self.device)

                            # local update
                            # print("local update for traj {}...".format(traj_i))
                            with torch.no_grad():
                                pred_y_next, traj_rep = self.BASEModel(qry_traj_i, qry_len_pad_i, qry_length_i, qry_time_up_i, qry_time_low_i, qry_time_up_diff_i,
                                                                       qry_time_low_diff_i, qry_dis_up_i, qry_dis_low_i, qry_dis_up_diff_i, qry_dis_low_diff_i)
                                # print("pred_y_next: ", pred_y_next)
                                # print("spt_next_truth: ", spt_next_truth_i)
                                Pres.append(pred_y_next)
                                Reps.append(traj_rep)
                    qry_Pres.append(Pres)
                    qry_Reps.append(Reps)

                traj_predict_loss = self.traj_level_loss(qry_Pres[0][0], qry_next_truth_i.long())
                traj_contrast_loss = self.TrajContrastiveLoss(qry_Reps[0][0], qry_Reps[1], qry_Reps[2])

                traj_loss = traj_predict_loss + traj_contrast_loss

                pred_probs = qry_Pres[0][0][0].tolist()
                descend_ranki = np.argsort(pred_probs)[::-1]

                pred_ranki = descend_ranki.tolist().index(qry_next_truth_i.item()) + 1
                traj_level_acc.append(pred_ranki)
                traj_level_losses.append(traj_loss)

        return traj_level_losses, traj_level_acc

# original localvalid
#     def localvalid(self, spt_all_info, qry_all_info, meta_model1, task_id_sameo_data):
#         traj_level_losses = []
#         traj_level_acc = []
#
#         # q_pred_y_next_list = []  # store all the prediction results (sorted prediction probabilities)
#         for up_step in range(self.update_step_test):
#
#             for traj_i in range(len(spt_all_info)):
#                 spt_info = spt_all_info[traj_i]
#                 qry_info = qry_all_info[traj_i]
#
#                 spt_input = data_to_device(spt_info)
#                 qry_input = data_to_device(qry_info)
#
#                 spt_traj, spt_next_truth, spt_des_truth, spt_len_pad, spt_length, \
#                 spt_time_up, spt_time_low, spt_time_up_diff, spt_time_low_diff, \
#                 spt_dis_up, spt_dis_low, spt_dis_up_diff, spt_dis_low_diff = spt_input
#
#                 qry_traj, qry_next_truth, qry_des_truth, qry_len_pad, qry_length, \
#                 qry_time_up, qry_time_low, qry_time_up_diff, qry_time_low_diff, \
#                 qry_dis_up, qry_dis_low, qry_dis_up_diff, qry_dis_low_diff = qry_input
#
#                 # Perform TrajDataLoader
#                 SPT_DATA = TrajDataLoader(spt_traj, spt_len_pad, spt_length, spt_time_up, spt_time_low, spt_time_up_diff,
#                                           spt_time_low_diff, spt_dis_up, spt_dis_low, spt_dis_up_diff, spt_dis_low_diff, spt_next_truth, spt_des_truth)
#
#                 QRY_DATA = TrajDataLoader(qry_traj, qry_len_pad, qry_length, qry_time_up, qry_time_low, qry_time_up_diff,
#                                           qry_time_low_diff, qry_dis_up, qry_dis_low, qry_dis_up_diff, qry_dis_low_diff, qry_next_truth, qry_des_truth)
#
#                 SPT_DATA_LOADER = DataLoader(SPT_DATA, batch_size=1)
#                 QRY_DATA_LOADER = DataLoader(QRY_DATA, batch_size=1)
#
#                 # local adapt
#                 print("local valid for traj {}...".format(traj_i))
#                 for spt_batch_i, (spt_traj_i, spt_len_pad_i, spt_length_i, spt_time_up_i, spt_time_low_i, spt_time_up_diff_i, spt_time_low_diff_i,
#                                   spt_dis_up_i, spt_dis_low_i, spt_dis_up_diff_i, spt_dis_low_diff_i, spt_next_truth_i, spt_des_truth_i) in enumerate(SPT_DATA_LOADER):
#                     pred_y_next = meta_model1(spt_traj_i, spt_len_pad_i, spt_length_i, spt_time_up_i, spt_time_low_i, spt_time_up_diff_i,
#                                                  spt_time_low_diff_i, spt_dis_up_i, spt_dis_low_i, spt_dis_up_diff_i, spt_dis_low_diff_i)
#
#                     traj_loss = self.traj_level_loss(pred_y_next, spt_next_truth_i.long())
#
#                     # check the correctness
#                     meta_model1.zero_grad()
#                     grad = torch.autograd.grad(traj_loss, meta_model1.parameters(), allow_unused=True)  # , create_graph=True
#
#                 # print("grad type:", type(grad))
#                 # param -= self.update_lr * grad
#                 #
#                 # traj_loss.backward(retain_graph=True)
#                 # with torch.no_grad():
#                 # for param in meta_model1.parameters():
#                 #     param_copy = param.detach().clone()
#                 #     if param.grad is not None:
#                 #         param_copy = param_copy - self.update_lr * param.grad
#                 #     else:
#                 #         print("param.grad is None.")
#                 #         continue
#                 #     param.data.copy_(param_copy)
#
#                 with torch.no_grad():
#                     i = 0
#                     for param in meta_model1.parameters():
#                         param_copy = param.clone()
#                         # print("param_copy", param_copy)
#                         # print("self.update_lr:", self.update_lr)
#                         param_copy = param_copy - self.update_lr * grad[i]
#                         param.data.copy_(param_copy)
#                         i += 1
#
#
#                 for qry_batch_i, (qry_traj_i, qry_len_pad_i, qry_length_i, qry_time_up_i, qry_time_low_i, qry_time_up_diff_i,
#                 qry_time_low_diff_i, qry_dis_up_i, qry_dis_low_i, qry_dis_up_diff_i, qry_dis_low_diff_i, qry_next_truth_i,
#                 qry_des_truth_i) in enumerate(QRY_DATA_LOADER):
#                     with torch.no_grad():
#                         q_pred_y_next = meta_model1(qry_traj_i, qry_len_pad_i, qry_length_i, qry_time_up_i,
#                                                     qry_time_low_i, qry_time_up_diff_i,
#                                                     qry_time_low_diff_i, qry_dis_up_i, qry_dis_low_i, qry_dis_up_diff_i,
#                                                     qry_dis_low_diff_i)
#
#                         traj_loss = self.traj_level_loss(q_pred_y_next, qry_next_truth_i.long())
#
#                 # get traj level acc
#                 pred_probs = q_pred_y_next[0].tolist()
#                 # get the index by item descending order
#                 descend_ranki = np.argsort(pred_probs)[::-1]
#
                # smaller pred_ranki, more accurate prediction
                # pred_ranki = descend_ranki.tolist().index(qry_next_truth_i.item()) + 1
                # traj_level_acc.append(pred_ranki)
                # traj_level_losses.append(traj_loss)
#
#         return traj_level_losses, traj_level_acc

