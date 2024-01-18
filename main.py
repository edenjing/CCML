# -*- coding: UTF-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import argparse
import gc
import pickle
import logging
import torch
import json
import time
import numpy as np
from copy import deepcopy
from collections import Counter
from torch import optim
from config import config_settings
from model.meta import Meta
from utils.metrics import Metrics
from utils.metadataset import TrainGenerator, fetch_sameo_data

logging.basicConfig(format='%(asctime)s - %(levelname)s -   '
                           '%(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
ARG = argparse.ArgumentParser()
parser = ARG
ARG = parser.parse_args()

root_path = './data/'
meta_path = root_path + 'dataset/final/'
embed_path = root_path + 'dataset/final/embeds/'
his_path = root_path + 'his_porto/'


def filter_statedict(module):
    # keep_vars=True: let the tensor variables in the state dictionary retain their computational graph,
    # which will be useful if we want to perform further computations on them later.
    state_dict = module.state_dict(keep_vars=True)
    non_params = []
    for key, value in state_dict.items():
        if not value.requires_grad:
            non_params.append(key)
    state_dict = module.state_dict()
    for key in non_params:
        del state_dict[key]
    return state_dict


def get_users(root_path, which="base"):
    config_path = root_path + 'dataset/config/'
    users_file = config_path + which + "_users.txt"
    users = []
    with open(users_file, "r") as f:
        for line in f:
            user = line.strip()
            users.append(user)
    return users


# instantiate: v. 例示，举例说明
def get_curriculum_user_ids(root_path):
    """
    TODO: One has to instantiate the curriculum according to the paper.
    That means to
        1. train the base model (STLSTM) of each base user independently;
        2. save the validation scores to the log;
        3. rank the users from easiest to hardest according to the scores and save the ranked user indices
            as a numpy array to "base_task_hardness.pkl". E.g., [4, 2, 1, 0, 5, 3, 7, 6] (max index should be the number of base users - 1).
    """
    # number of curriculum = number of base users
    mtrain_user_ids = get_users(root_path, 'base')
    # mtrain_user_num = len(mtrain_user_ids)
    # return list(np.arange(mtrain_user_num))
    return mtrain_user_ids


def get_model(root_path, config, model_name="Meta"):
    # update some paths to config
    model_save_path = root_path + "model_save/"
    loss_save_path = root_path + "loss_save/"

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    if not os.path.exists(loss_save_path):
        os.mkdir(loss_save_path)

    config['save_path'] = "{}.pt".format(model_save_path + model_name)
    config['loss_save_path'] = "{}loss_{}.txt".format(loss_save_path, model_name)

    # create model
    # if model_name == "Meta":
    #     model = None
    meta_model = Meta(config)
    return meta_model


def get_optimizer(meta_model, config):
    # The result list contains only the "trainable" parameters of meta_model.net
    # as only the trainable parameters need to be initialized
    init_parameters = list(filter(lambda p: p.requires_grad, meta_model.BASEModel.parameters()))
    parameters = [
        {'params': init_parameters, 'lr': config['meta_lr']},
    ]
    optimizer = optim.Adam(parameters, lr=config['meta_lr'], eps=1e-3)
    # learning rate scheduler: adjust the learning rate used by the optimizer
    # This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs,
    # the learning rate is reduced.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=PATIENCE,
                                                        verbose=True, min_lr=1e-6)
    return optimizer, scheduler


# embed_path = root_path + 'dataset/final/embeds/'
# get embeddings of location, time slot and distance slot
def get_embeds(embed_path):
    loc_emb_file = "{}loc_embed.npy".format(embed_path)
    tslot_emb_file = "{}tslot_embed.npy".format(embed_path)
    sslot_emb_file = "{}sslot_embed.npy".format(embed_path)

    loc_emb = torch.from_numpy(np.load(loc_emb_file).astype(np.float32))
    tslot_emb = torch.from_numpy(np.load(tslot_emb_file).astype(np.float32))
    sslot_emb = torch.from_numpy(np.load(sslot_emb_file).astype(np.float32))
    embeds = [loc_emb, tslot_emb, sslot_emb]
    return embeds


# Output:
# task_idx2acc: task id -> task level accuracy
# task_idx_to_user2acc: task id -> user id -> user level accuracy
def update_hardness(task_ids, task_sample_id2traj_id, acc_results):
    task_level_acc = acc_results["task_level_acc"]
    task_traj_level_acc = acc_results["task_traj_level_acc"]

    task_id2acc = {}
    task_id_to_traj2acc = {}
    for i in range(len(task_ids)):
        # 1) for [hard-task] sampling
        task_id2acc[task_ids[i]] = task_level_acc[i]
        # 2) for [hard-traj] sampling
        traj_level_acc = task_traj_level_acc[i]  # traj -> acc
        sample_sub2traj = task_sample_id2traj_id[i]  # traj ids
        # print("sample_sub2traj: ", sample_sub2traj)

        traj2acc = {}
        for j in range(len(traj_level_acc)):
            traj_id = sample_sub2traj[j]
            traj2acc[traj_id] = traj_level_acc[j]

        # smaller rank of a traj, more accurate the prediction
        traj2acc = dict(sorted(traj2acc.items(), key=lambda x: x[1]))
        task_id_to_traj2acc[task_ids[i]] = traj2acc

    # larger count number, more accurate the prediction
    task_id2acc = dict(sorted(task_id2acc.items(), key=lambda x: x[1], reverse=True))

    task_id2results = {"task_id2acc": task_id2acc,
                       "task_id_to_traj2acc": task_id_to_traj2acc,}

    return task_id2results


def one_meta_training_step(task_gen, sameo_mtrain_tasks, meta_model, users_wp, optimizer, update_hardness_count, device, parameters, task_id2results, stage, curriculum, hard_task, train_step):
    # fetch_task_batch(): fetch task -> traj samples
    # data[task_id] = [spt_sample_all_information, qry_sample_all_information]
    # task_ids -> a list A, stores user ids in a task batch
    # task_sample_id2traj_id -> a 2D list B, each B[i] stores the selected traj indexes for user(task) A[i]

    # fetch generated task data in current iteration
    data, task_ids, task_sample_id2traj_id = task_gen.fetch_task_batch(
        task_id2results=task_id2results, stage=stage, curriculum=curriculum, hard_task=hard_task, train_step=train_step,
    )
    # fetch generated sameo task_data in current iteration
    sameo_tasks_data = fetch_sameo_data(sameo_mtrain_tasks, task_ids)

    # data, init_embeds = task_to_device(*data, init_embeds, device)
    # print("Training Start...")
    loss, acc_results = meta_model(
        data, sameo_tasks_data, task_ids, users_wp, optimizer, parameters)

    print("total_loss: ", loss.item())

    # print("accs (in one_meta_training_step): ", accs)
    # print("loss_q (in one_meta_training_step): ", loss_q)
    # print("results (in one_meta_training_step): ", results)

    # optimizer.zero_grad()
    # print("loss in one_meta_training_step: ", loss.item())
    # loss.backward()
    '''
    torch.nn.utils.clip_grad_value_(parameters, clip_value)
    # Clip gradient of an iterable of parameters at specified value.
    # parameters: an iterable of Tensors or a single Tensor that will have gradients normalized;
    # clip_value: maximum allowed value of the gradients, the gradients are clipped in the range [-clip_value, clip_value];
    '''
    # torch.nn.utils.clip_grad_value_(parameters, 0.25)
    # optimizer.step()

    update_hardness_count += 1

    # acc_results, loss_q.item() are for logging during training, task_id2results supports the next meta training step
    task_id2results = update_hardness(task_ids, task_sample_id2traj_id, acc_results)
    return acc_results, loss.item(), task_id2results, update_hardness_count


def main_meta(meta_path, root_path, embed_path, model_name="Meta"):
    # read config file
    config = config_settings
    print(config)

    # get meta_model, optimizer, metrics
    # meta_model, model = get_model(root_path, config, model_name)
    # meta_model = get_model(root_path, config, model_name)

    model_save_path = root_path + "model_save/"
    loss_save_path = root_path + "loss_save/"

    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    if not os.path.exists(loss_save_path):
        os.mkdir(loss_save_path)

    config['save_path'] = "{}.pt".format(model_save_path + model_name)
    config['loss_save_path'] = "{}loss_{}.txt".format(loss_save_path, model_name)

    # create model
    # if model_name == "Meta":
    #     model = None
    meta_model = Meta(config)

    # print("meta_model:", meta_model)

    optimizer, scheduler = get_optimizer(meta_model, config)
    init_embeds = get_embeds(embed_path)

    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta_model = meta_model.to(device)
    # if model is not None:
    #     model.to(device)

    # Calculate the total number of trainable parameters in the model
    # that have their "requires_grad" attribute set to "True".
    parameters = list(filter(lambda p: p.requires_grad, meta_model.parameters()))
    tmp = filter(lambda x: x.requires_grad, meta_model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(meta_model)
    logger.info('Total trainable tensors: {}'.format(num))
    metric = Metrics()

    # load the train/valid/test dataset
    # Format: user id -> [spt info, qry info]
    mtrain_tasks = pickle.load(open(meta_path + "mtrain_tasks.pkl", 'rb'))
    mvalid_tasks = pickle.load(open(meta_path + "mvalid_tasks.pkl", 'rb'))
    mtest_tasks = pickle.load(open(meta_path + "mtest_tasks.pkl", 'rb'))
    logger.info('Loaded all the data pickles!')

    # load the sameo_train/valid/test dataset dataset
    # Format: user id -> [spt info, qry info]
    # spt/qry info: [same_o_trajs, same_o_times, same_o_lengths]
    sameo_mtrain_tasks = pickle.load(open(meta_path + "sameo_mtrain_tasks.pkl", 'rb'))
    sameo_mvalid_tasks = pickle.load(open(meta_path + "sameo_mvalid_tasks.pkl", 'rb'))
    sameo_mtest_tasks = pickle.load(open(meta_path + "sameo_mtest_tasks.pkl", 'rb'))
    logger.info('Loaded all the sameo data pickles!')

    # load user spatial distribution file
    with open(his_path + 'user_wp.json', 'r') as f1:
        users_wp = json.load(f1)  # all users spatial distribution

    # set variables for statistics
    best_scores = 0
    running_loss = 0
    train_step = 0

    update_hardness_count = 0

    # start training
    running_accs = np.zeros(config['update_step'] + 1)
    task_gen = TrainGenerator(mtrain_tasks, config['task_batch_size'],
                              curriculum_user_ids=get_curriculum_user_ids(root_path), pacing_function=PACING_FUNCTION,
                              few_traj_num=config['few_traj_num'], max_steps=config['max_train_steps'])  # config['train_qry_batch_size'],

    task_id2results = {"task_id2acc": None,
                        "task_id_to_traj2acc": None}

    hard_task_counter = Counter()

    while True:
        # >>>>> [Stage1]: sample the hardest tasks of last round, and then sample more tasks.
        # >>>>> Here the "task" means "user".
        print("=== Train Tasks ===")
        print("[Stage1]")
        acc_results, loss, task_id2results, update_hardness_count = one_meta_training_step(
            task_gen, sameo_mtrain_tasks, meta_model, users_wp, optimizer, update_hardness_count, device, parameters,
            task_id2results, "stage1", CURRICULUM, HARD_USER, train_step=train_step,
        )
        print("Hardness Update Count: ", update_hardness_count)

        # loss: mean value of task level loss -> mean value of traj level loss
        running_loss += loss

        # task_level_acc: for each task, the count of traj prediction result that is within top-k
        running_accs += np.mean(np.array(acc_results["task_level_acc"]))
        train_step += 1
        print("train_step (in stage 1): ", train_step)
        hard_task_counter.update(list(task_id2results["task_id2acc"].keys()))

        if HARD_TRAJ:
            # >>>>> [Stage2]: the same tasks as Stage 1, keep the hardest users, and sample new users in these tasks.
            # >>>>> Here the "task" means "user".
            print("=== Train Tasks ===")
            print("[Stage2]")
            acc_results, loss, task_id2results, update_hardness_count = one_meta_training_step(
                task_gen, sameo_mtrain_tasks, meta_model, users_wp, optimizer, update_hardness_count, device, parameters,
                task_id2results, "stage2", CURRICULUM, HARD_USER, train_step=train_step,
            )
            print("Hardness Update Count: ", update_hardness_count)

            running_loss += loss
            running_accs += np.mean(np.array(acc_results["task_level_acc"]))
            train_step += 1
            print("train_step (in stage 2): ", train_step)
            hard_task_counter.update(list(task_id2results["task_id2acc"].keys()))

        # stage1和stage2是平级的

        if train_step > config['max_train_steps']:
            print("train_step > config['max_train_steps']: ", train_step)
            break

        # compute the training loss per PER_TRAIN_LOG train step
        if (train_step / STAGE_NUM + 1) % PER_TRAIN_LOG == 0:
            print("train_step: ", train_step)
            print("PER_TRAIN_LOG: ", PER_TRAIN_LOG)
            print("train_step / STAGE_NUM + 1: ", train_step / STAGE_NUM + 1)
            # print("(train_step / STAGE_NUM + 1) % PER_TRAIN_LOG: ", (train_step / STAGE_NUM + 1) % PER_TRAIN_LOG)

            training_loss = running_loss / PER_TRAIN_LOG / STAGE_NUM
            print('Task Train Step[{}]: loss_q: {:.5f}, training accs: {}'.format(train_step + 1, training_loss,
                                                                             running_accs / PER_TRAIN_LOG / STAGE_NUM))
            running_loss = 0
            running_accs = np.zeros(config['update_step'] + 1)

        # perform model validation/test per PER_TEST_LOG train step
        if (train_step / STAGE_NUM + 1) % PER_TEST_LOG == 0:
            print("train_step: ", train_step)
            print("PER_TEST_LOG: ", PER_TEST_LOG)
            print("train_step / STAGE_NUM + 1: ", train_step / STAGE_NUM + 1)
            # print("(train_step / STAGE_NUM + 1) % PER_TEST_LOG: ", (train_step / STAGE_NUM + 1) % PER_TEST_LOG)

            torch.cuda.empty_cache()
            gc.collect()
            meta_model.eval()
            # print(meta_model)

            print("=== Valid Tasks ===")
            total_loss, valid_scores, valid_score_weights = meta_model.evaluate(mvalid_tasks, sameo_mvalid_tasks, metric, users_wp, test=False,
                                                                                few_task_num=config['valid_task_num'], few_traj_num=config['few_traj_num'])
            # task_level_acc: store the number of trajs in each task that acc (rank) is within top_k
            # valid_task_acc = np.mean(np.array(task_level_acc))
            print("valid task num: ", len(valid_score_weights), "valid total loss: {:.5f}".format(total_loss.item()))  #, "valid task level acc:", valid_task_acc)

            avg_valid_scores = np.average(np.array(valid_scores), axis=0, weights=np.array(valid_score_weights))
            print("Average valid scores: {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(*avg_valid_scores))

            # Efficiency Evaluation

            test_start_time = time.time()

            print("=== Test Tasks ===")
            test_scores, test_score_weights = meta_model.evaluate(mtest_tasks, sameo_mtest_tasks, metric, users_wp, test=True,
                                                                  few_task_num=config['test_task_num'], few_traj_num=config['few_traj_num'])

            test_end_time = time.time()
            testing_time = test_end_time - test_start_time
            # avg_user_time = testing_time / len(mtest_tasks)
            # print("testing time (user): ", avg_user_time)
            print("total testing time: ", testing_time)

            # task_level_acc: store the number of trajs in each task that acc (rank) is within top_k
            # test_task_acc = np.mean(np.array(task_level_acc))
            print("test task num: ", len(test_score_weights))  #, "test task level acc:", test_task_acc)

            avg_test_scores = np.average(np.array(test_scores), axis=0, weights=np.array(test_score_weights))
            print("Average test scores: {:.5f}\t{:.5f}\t{:.5f}\t{:.5f}".format(*avg_test_scores))

            print("====================")
            valid_tot_score = np.mean(avg_valid_scores)
            # valid_tot_score = avg_valid_scores
            print("valid_tot_score: ", valid_tot_score)

            if valid_tot_score < best_scores:
                best_scores = valid_tot_score
                print("best_scores: ", best_scores)
                dict_save_path = os.path.join(meta_path + "model_save/", str(train_step + 1) + ".dict")
                torch.save(filter_statedict(meta_model), dict_save_path)
                logger.info("Best metrics: {}! Save model to {}".format(valid_tot_score, dict_save_path))

            '''
            # scheduler.step(): update the learning rate based on the current epoch's performance and return the new learning rate
            # If we don't call the scheduler.step() method, the learning rate will not change, and the model will continue to use the initial learning rate throughout the training,
            # which may result in suboptimal performance, longer training times, and lower accuracy of the model
            '''
            '''
            # Why the valid_tot_score need to be passed by?
            # Because we adopt the ReduceLROnPlateau scheduler which updates the learning rate when the validation loss stops improving, 
            # so we need to specify the validation loss as the argument to scheduler.step()
            '''

            # Note that step should be called after validation
            scheduler.step(valid_tot_score)

            meta_model.train()
            print("hard_task_counter:", hard_task_counter)
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # Settings for CMCTP
    CURRICULUM = True
    HARD_USER = True  # equal to hard task (original description)
    HARD_TRAJ = True
    USER_LEVEL_CL = True
    TRAJ_LEVEL_CL = True

    # ssp: single step pacing - 选取user sampling pool内的user时，用到的single step strategy
    PACING_FUNCTION = 'ssp'
    if HARD_TRAJ:
        STAGE_NUM = 2
    else:
        STAGE_NUM = 1

    # Regardless of stage 1 or 2，compute training loss per 5 iterations perform model evaluation per 5 iterations
    PER_TRAIN_LOG = 5 // STAGE_NUM
    PER_TEST_LOG = 5 // STAGE_NUM

    PATIENCE = 2  # patience for learning rate
    INIT_COMPARE = False

    logger.info(
        "HARD_USER: {}, HARD_TRAJ: {}, CURRICULUM: {}, PACING_FUNCTION: {}, PER_TRAIN_LOG：{}, PER_TEST_LOG: {}, PATIENCE: {}, USER_LEVEL_CL: {}, TRAJ_LEVEL_CL:{}".format(
            HARD_USER, HARD_TRAJ, CURRICULUM, PACING_FUNCTION, PER_TRAIN_LOG, PER_TEST_LOG, PATIENCE, USER_LEVEL_CL, TRAJ_LEVEL_CL))
    logger.info("curriculum is: {}".format(get_curriculum_user_ids(root_path)))

    main_meta(meta_path, root_path, embed_path)
