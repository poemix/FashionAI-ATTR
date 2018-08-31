# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import os
import cv2
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


def cam_writer(save_path, queue, stop_token='stop'):
    """Write cam to npy file."""
    while True:
        token, name_batch, cam_batch = queue.get()
        if token == stop_token:
            return
        for idx, name in enumerate(name_batch):
            path = '{}/{}'.format(save_path, '/'.join(name.split('/')[0:-1]))
            if not os.path.exists(path):
                os.makedirs(path)
            np.save('{}/{}.npy'.format(path, name.split('/')[-1].split('.')[0]), cam_batch[idx])


def roi_writer(save_path, queue, stop_token='stop'):
    """Write ROI image to jpg file."""
    while True:
        token, root, name_batch, cam_batch = queue.get()
        if token == stop_token:
            return
        for idx, name in enumerate(name_batch):
            print(idx, name)
            cam = cam_batch[idx]
            path = '{}/{}'.format(save_path, '/'.join(name.split('/')[0:-1]))
            if not os.path.exists(path):
                os.makedirs(path)
            img = cv2.imread('{}/{}'.format(root, name), cv2.IMREAD_UNCHANGED)
            cam = cv2.resize((cam * 255).astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
            heatmap[cam <= hp.threshold] = 0

            gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, hp.threshold, 255, cv2.THRESH_BINARY)
            pos = np.where(thresh == 255)
            x, y, w, h = cv2.boundingRect(np.vstack((pos[1], pos[0])).T)
            cv2.imwrite('{}/{}'.format(save_path, name), img[y: y + h, x: x + w])


def vis_cam(root, names, cams):
    for idx, cam in enumerate(cams):
        name = names[idx]
        print(idx, name)
        img = cv2.imread('{}/{}'.format(root, name), cv2.IMREAD_UNCHANGED)
        cam = cv2.resize((cam * 255).astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap[cam <= hp.threshold] = 0

        gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, hp.threshold, 255, cv2.THRESH_BINARY)
        pos = np.where(thresh == 255)
        x, y, w, h = cv2.boundingRect(np.vstack((pos[1], pos[0])).T)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out = cv2.addWeighted(img, 0.8, heatmap, 0.4, 0)
        cv2.imshow('img', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
