config_settings = {
    # Parameters for CMCTP
    "update_step": 1,  # task-level inner update steps
    "update_step_test": 1,  # update steps for fine tuning / local valid
    # "train_qry_batch_size": 10,
    "task_batch_size": 3,  # user/task num in each batch/iteration, max num: 250

    "max_train_steps": 100000,

    "meta_lr": 1e-5,
    "update_lr": 1e-5,
    "loss_coef": 1e-3,
    # "local_fix_var": 1,
    # "sample_batch_size": 1024,

    "valid_task_num": 40,  # max valid task num: 84
    "test_task_num": 50,  # max test task num: 100
    "few_traj_num": 100,  # traj num for each user in training step

    # Parameters for Metric
    "top_k": 5,

    # Parameters for Base Model
    'd': 12,
    'd1': 12,
    "hidden_neurons": 100,

    # Parameters for user-level CL
    'num_clusters': 3,
    'user_contrast_margin': 1.0,

    # Parameters for traj-level CL
    'temperature': 1.0,

    'mask_bool': True,
    'truncate_bool': True,
    'two_hop_bool': True,
    'detour_bool': True,

    'mask_ratio': 0.2,
    'truncate_ratio': 0.3,
    'replace_ratio': 0.3,
}
