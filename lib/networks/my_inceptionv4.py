# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.networks.slim.nets import inception_v4
from experiment.hyperparams import HyperParams as hp


def run_proc(sess, reader, variables, ignore):
    for name in variables:
        try:
            tensor = sess.graph.get_tensor_by_name('{}/{}:0'.format('MyInceptionV4', name))
            weight = reader.get_tensor(name)
            sess.run(tf.assign(tensor, weight))
            print('{} assign success.'.format(name))
        except KeyError as e:
            if ignore:
                print(e, name)
            else:
                print(e)
                raise e
        except Exception as e:
            print(e, name)
            raise e


class MyInceptionV4(object):
    def __init__(self, data, label, keep_prob, is_training):
        # placeholder
        self.data = data
        self.label = label
        self.keep_prob = keep_prob
        self.is_training = is_training

        self.layers = dict(data=data, label=label)
        self.setup()

    def setup(self):
        with tf.variable_scope(self.__class__.__name__):
            with slim.arg_scope(inception_v4.inception_v4_arg_scope()):
                net, end_points = inception_v4.inception_v4(
                    self.data,
                    num_classes=None,
                    create_aux_logits=False,
                    is_training=self.is_training)
                self.layers['PreGAP'] = end_points['PreGAP']

        with tf.variable_scope('{}/logits'.format(self.__class__.__name__)):
            net = tf.squeeze(net, axis=[1, 2])
            net = tf.nn.dropout(net, keep_prob=self.keep_prob)
            cls_score_coat = slim.fully_connected(net, 8*2, activation_fn=None, scope='coat_length_labels')
            cls_prob_coat = slim.softmax(cls_score_coat, scope='coat_length_labels')

            self.layers['cls_score_coat_length_labels'] = cls_score_coat
            self.layers['cls_prob_coat_length_labels'] = cls_prob_coat

            cls_score_collar = slim.fully_connected(net, 5, activation_fn=None, scope='collar_design_labels')
            cls_prob_collar = slim.softmax(cls_score_collar, scope='collar_design_labels')

            self.layers['cls_score_collar_design_labels'] = cls_score_collar
            self.layers['cls_prob_collar_design_labels'] = cls_prob_collar

            cls_score_lapel = slim.fully_connected(net, 5, activation_fn=None, scope='lapel_design_labels')
            cls_prob_lapel = slim.softmax(cls_score_lapel, scope='lapel_design_labels')

            self.layers['cls_score_lapel_design_labels'] = cls_score_lapel
            self.layers['cls_prob_lapel_design_labels'] = cls_prob_lapel

            cls_score_neck = slim.fully_connected(net, 5, activation_fn=None, scope='neck_design_labels')
            cls_prob_neck = slim.softmax(cls_score_neck, scope='neck_design_labels')

            self.layers['cls_score_neck_design_labels'] = cls_score_neck
            self.layers['cls_prob_neck_design_labels'] = cls_prob_neck

            cls_score_neckline = slim.fully_connected(net, 10, activation_fn=None, scope='neckline_design_labels')
            cls_prob_neckline = slim.softmax(cls_score_neckline, scope='neckline_design_labels')

            self.layers['cls_score_neckline_design_labels'] = cls_score_neckline
            self.layers['cls_prob_neckline_design_labels'] = cls_prob_neckline

            cls_score_pant = slim.fully_connected(net, 6 * 2, activation_fn=None, scope='pant_length_labels')
            cls_prob_pant = slim.softmax(cls_score_pant, scope='pant_length_labels')

            self.layers['cls_score_pant_length_labels'] = cls_score_pant
            self.layers['cls_prob_pant_length_labels'] = cls_prob_pant

            cls_score_skirt = slim.fully_connected(net, 6 * 2, activation_fn=None, scope='skirt_length_labels')
            cls_prob_skirt = slim.softmax(cls_score_skirt, scope='skirt_length_labels')

            self.layers['cls_score_skirt_length_labels'] = cls_score_skirt
            self.layers['cls_prob_skirt_length_labels'] = cls_prob_skirt

            cls_score_sleeve = slim.fully_connected(net, 9 * 2, activation_fn=None, scope='sleeve_length_labels')
            cls_prob_sleeve = slim.softmax(cls_score_sleeve, scope='sleeve_length_labels')

            self.layers['cls_score_sleeve_length_labels'] = cls_score_sleeve
            self.layers['cls_prob_sleeve_length_labels'] = cls_prob_sleeve

    def save(self, sess, saver, step, dir=None):
        if dir is None:
            output_dir = hp.output_dir.format(hp.pro_path, self.__class__.__name__)
        else:
            output_dir = hp.output_dir.format(hp.pro_path, dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        infix = ('_' + hp.snapshot_infix if hp.snapshot_infix != '' else '')
        filename = (hp.snapshot_prefix + infix + '_iter_{:d}'.format(step + 1) + '.ckpt')
        filename = os.path.join(output_dir, filename)

        saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    def load(self, sess, file, ignore=True):
        reader = tf.train.NewCheckpointReader(file)
        variables = reader.get_variable_to_shape_map()
        run_proc(sess, reader, variables, ignore)


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 331, 331, 3])
    y = tf.placeholder(tf.float32, [None, None])
    dropout = tf.placeholder(tf.float32)
    phase = tf.placeholder(tf.bool)
    network = MyInceptionV4(x, y, dropout, phase)
