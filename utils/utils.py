# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from experiment.data_info import DataInfo as di
from experiment.hyperparams import HyperParams as hp


def session(graph=None, allow_soft_placement=hp.allow_soft_placement, log_device_placement=hp.log_device_placement,
            allow_growth=hp.allow_growth):
    """return a session with simple config."""
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)


def preprocess_label(label, n_sample, n_class):
    Y = np.zeros(n_sample, dtype=np.float32)
    for i, value in enumerate(label):
        for j in range(n_class):
            if value[j] == 'y':
                Y[i] = j
    return Y


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].
    Args:
      x: input Tensor.
      func: Python function to apply.
      num_cases: Python int32, number of cases to sample sel from.
    Returns:
      The result of func(x, sel), where func receives the value of the
      selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge(
        [func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case) for case in range(num_cases)]
    )[0]


def csv_writer(save_path, queue, stop_token='stop'):
    """Write predict prob to csv file."""
    df = pd.DataFrame()
    names = []
    attr_keys = []
    probs = []
    while True:
        token, name_batch, attr_key, prob_batch = queue.get()
        if token == stop_token:
            df['name'] = names
            df['attr_key'] = attr_keys
            df['prob'] = probs
            df.to_csv(save_path, index=False, header=False)
            return
        n_class = di.num_classes[attr_key]
        for idx, name in enumerate(name_batch):
            names.append(name)
            attr_keys.append(attr_key)
            prob = prob_batch[idx]
            k = np.argmax(prob)
            if k > n_class - 1:
                probs.append(';'.join(list(map(lambda x: '{:.4f}'.format(x), prob[n_class:]))))
            else:
                probs.append(';'.join(list(map(lambda x: '{:.4f}'.format(x), prob[:n_class]))))


def heatmap_writer(save_path, queue, stop_token='stop'):
    """Write heatmap to npy file."""
    pass


def roi_writer(save_path, queue, stop_token='stop'):
    """Write ROI image to jpg file."""
    pass
