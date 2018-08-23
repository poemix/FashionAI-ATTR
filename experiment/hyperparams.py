# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import os


class HyperParams(object):
    # project path
    pro_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")).replace('\\', '/')

    # random seed
    seed = 2018

    # optimizer
    learning_rate = 0.01
    stepsize = 5000
    lr_decay = 0.1
    momentum = 0.9
    keep_prob = 0.5

    num_epoch = 15
    snapshot_iter = 100
    display = 10

    # save model
    output_dir = '{}/model/{}'
    snapshot_infix = ''
    snapshot_prefix = 'fashion-ai'

    # data loader
    shuffle = True
    min_after_dequeue = 32
    allow_smaller_final_batch = False
    num_threads = 2
    batch_size = 12

    # session config
    allow_soft_placement = False
    log_device_placement = False
    allow_growth = True

    times = 5
    n_task = 8

    threshold = 64
