# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import datetime
import numpy as np
import tensorflow as tf
from lib.networks.my_inceptionv4 import MyInceptionV4
from lib.datasets import transforms
from lib.datasets.data_loader import MTCSVLoader
from experiment.hyperparams import HyperParams as hp
from utils import session


def train_net():
    # default graph
    # placeholder
    data = tf.placeholder(tf.float32, [None, 331, 331, 3])
    label = tf.placeholder(tf.float32, [None, None])
    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    # network
    net = MyInceptionV4(data=data, label=label, keep_prob=keep_prob, is_training=is_training)

    # batch data
    transformer = transforms.Sequential(
        [
            transforms.Resize([331, 331]),
            transforms.Preprocess(),
            transforms.RandomHorizontalFlip(),
        ]
    )

    loader = MTCSVLoader(root='E:/fashion-dataset/base', csv_path='{}/dataset/labels/s1_label.csv'.format(hp.pro_path),
                         batch_size=hp.batch_size, transformer_fn=transformer,
                         shuffle=hp.shuffle, min_after_dequeue=hp.min_after_dequeue,
                         num_threads=hp.num_threads, allow_smaller_final_batch=hp.allow_smaller_final_batch,
                         seed=hp.seed, num_epochs=None)

    num_batch = sum(list(map(lambda x: x // hp.batch_size, loader.n_sample.values()))) // hp.n_task
    hp.display = 1
    hp.snapshot_iter = num_batch
    hp.stepsize = num_batch * 5

    flat_logits = {}
    y_preds = {}
    y_trues = {}
    accuracys = {}
    losses = {}
    train_ops = {}
    flat_label = net.layers['label']

    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(hp.learning_rate, global_step, hp.stepsize, hp.lr_decay, staircase=True)
    opt1 = tf.train.MomentumOptimizer(lr, hp.momentum)
    opt2 = tf.train.MomentumOptimizer(hp.times * lr, hp.momentum)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_op = tf.group(*update_ops)

    var1 = tf.trainable_variables()[0:-16]
    var2 = tf.trainable_variables()[-16:]

    for attr_key in ['coat_length_labels', 'collar_design_labels', 'lapel_design_labels', 'neck_design_labels',
                     'neckline_design_labels', 'pant_length_labels', 'skirt_length_labels', 'sleeve_length_labels']:
        # pred
        cls_score = 'cls_score_{}'.format(attr_key)
        flat_logit = net.layers[cls_score]
        flat_logits[attr_key] = flat_logit

        # acc
        y_pred = tf.argmax(flat_logit, axis=1)
        y_true = tf.argmax(flat_label, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))

        y_preds[attr_key] = y_pred
        y_trues[attr_key] = y_true
        accuracys[attr_key] = accuracy

        # loss
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=flat_logit,
                labels=flat_label
            )
        )
        losses[attr_key] = loss

        with tf.control_dependencies([update_op]):
            cost = tf.identity(loss)
        # optimizer
        train_op1 = opt1.minimize(cost, global_step=global_step, var_list=var1)
        train_op2 = opt2.minimize(cost, var_list=var2)
        train_op = tf.group(train_op1, train_op2)
        train_ops[attr_key] = train_op

    # session
    sess = session()
    saver = tf.train.Saver(max_to_keep=50)
    sess.run(tf.global_variables_initializer())
    net.load(sess, '{}/model/Pretrain/{}'.format(hp.pro_path, 'inception_v4.ckpt'))
    step = -1
    last_snapshot_iter = -1
    max_step = hp.num_epoch * num_batch
    print('num_batch', num_batch)
    now_ime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('%s Start Training...iter: %d(%d) / %d(%d) lr: %s' % (now_ime, (step + 1), 0, max_step, hp.num_epoch, sess.run(lr)))

    AK = ['coat_length_labels', 'collar_design_labels', 'lapel_design_labels', 'neck_design_labels',
          'neckline_design_labels', 'pant_length_labels', 'skirt_length_labels', 'sleeve_length_labels']

    for epoch in range(hp.num_epoch):
        for batch in range((epoch * num_batch), ((epoch + 1) * num_batch)):
            np.random.shuffle(AK)
            for step in range((batch * hp.n_task), ((batch + 1) * hp.n_task)):
                j = step % hp.n_task
                ak = AK[j]
                print(ak)
                data_batch, label_batch, name_batch = loader.batch(ak)
                train_op = train_ops[ak]
                loss = losses[ak]
                y_pred = y_preds[ak]
                y_true = y_trues[ak]
                accuracy = accuracys[ak]

                _ = sess.run(
                    train_op,
                    feed_dict={
                        net.data: data_batch,
                        net.label: label_batch,
                        net.is_training: True,
                        net.keep_prob: hp.keep_prob
                    }
                )
                if (step + 1) % hp.display == 0:
                    loss_value, acc, y1, y2 = sess.run(
                        [loss, accuracy, y_true, y_pred],
                        feed_dict={
                            net.data: data_batch,
                            net.label: label_batch,
                            net.is_training: False,
                            net.keep_prob: 1.0
                        }
                    )
                    print(y1)
                    print(y2)
                    now_ime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print('%s iter: %d(%d) / %d(%d), total loss: %.4f, acc: %.2f, lr: %s'
                          % (now_ime, (step + 1), (epoch + 1), max_step, hp.num_epoch, loss_value, acc, sess.run(lr)))
                if (step + 1) % hp.snapshot_iter == 0:
                    last_snapshot_iter = step
                    net.save(sess, saver, step)

    if last_snapshot_iter != step:
        net.save(sess, saver, step)

    sess.close()


if __name__ == '__main__':
    train_net()
