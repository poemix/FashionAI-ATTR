# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import pandas as pd
import tensorflow as tf
from utils import session
from experiment.data_info import DataInfo as di


def csv_batch(root, data_path, label, n_class, batch_size, height=331, width=331, transformer_fn=None,
              num_epochs=None, shuffle=True, min_after_dequeue=50, allow_smaller_final_batch=False,
              num_threads=2, seed=None, scope=None):
    with tf.name_scope(scope, 'csv_batch'):
        n_sample = len(data_path)
        name = tf.convert_to_tensor(data_path, tf.string)
        data_path = tf.convert_to_tensor(list(map(lambda x: '{}{}'.format(root, x), data_path)), tf.string)
        return_label = True
        try:
            label = tf.convert_to_tensor(label, tf.int32)
        except Exception as e:
            print(e)
            label = tf.convert_to_tensor(label, tf.string)
            return_label = False

        data_path, label, name = tf.train.slice_input_producer([data_path, label, name],
                                                               shuffle=shuffle,
                                                               capacity=n_sample, seed=seed,
                                                               num_epochs=num_epochs)

        image_value = tf.read_file(data_path)
        data = tf.image.decode_jpeg(image_value, channels=3)

        if return_label:
            label = tf.one_hot(label, n_class, on_value=1., off_value=0.)

        if transformer_fn is not None:
            data = transformer_fn(data)
        else:
            data = tf.image.resize_images(data, [height, width])

        if shuffle:
            capacity = min_after_dequeue + (num_threads + 1) * batch_size
            data_batch, label_batch, name_batch = tf.train.shuffle_batch([data, label, name],
                                                                         batch_size=batch_size,
                                                                         capacity=capacity,
                                                                         min_after_dequeue=min_after_dequeue,
                                                                         num_threads=num_threads,
                                                                         allow_smaller_final_batch=allow_smaller_final_batch,
                                                                         seed=seed)
        else:
            capacity = (num_threads + 1) * batch_size
            data_batch, label_batch, name_batch = tf.train.batch([data, label, name],
                                                                 batch_size=batch_size,
                                                                 capacity=capacity,
                                                                 allow_smaller_final_batch=allow_smaller_final_batch)

        return [data_batch, label_batch, name_batch], n_sample


class MTCSVLoader(object):
    def __init__(self, root, csv_path, batch_size, height=331, width=331, transformer_fn=None, num_epochs=None,
                 shuffle=True, min_after_dequeue=25, allow_smaller_final_batch=False, num_threads=2, seed=None):
        root = root.replace('\\', '/')
        root = root if root[-1] == '/' else '{}/'.format(root)
        self.root = root
        self.csv_path = csv_path.replace('\\', '/')
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.transformer_fn = transformer_fn
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.min_after_dequeue = min_after_dequeue
        self.allow_smaller_final_batch = allow_smaller_final_batch
        self.num_threads = num_threads
        self.seed = seed

        df = pd.read_csv(self.csv_path, header=None)

        if shuffle:
            df = df.sample(frac=1., random_state=seed)

        try:
            df[2] = df.apply(lambda line: line[2].index('y') + len(line[2]) if line[3] else line[2].index('y'), axis=1)
        except Exception as e:
            print(e)

        print('{}: create session!'.format(self.__class__.__name__))
        self.batch_ops = {}
        self.n_sample = {}
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/cpu:0'):
                for attr_key, n_class in di.num_classes_v2.items():
                    data_path = df[df[1].isin([attr_key])][0].values
                    label = df[df[1].isin([attr_key])][2].values

                    batch_ops, n_sample = csv_batch(root=self.root, data_path=data_path, label=label,
                                                    n_class=n_class, batch_size=batch_size,
                                                    height=height, width=width,
                                                    transformer_fn=transformer_fn, num_epochs=num_epochs,
                                                    shuffle=shuffle, min_after_dequeue=min_after_dequeue,
                                                    allow_smaller_final_batch=allow_smaller_final_batch,
                                                    num_threads=num_threads, seed=seed)
                    self.batch_ops[attr_key] = batch_ops
                    self.n_sample[attr_key] = n_sample
                if num_epochs is not None:
                    self.init = tf.local_variables_initializer()
        self.sess = session(graph=self.graph)
        if num_epochs is not None:
            self.sess.run(self.init)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def batch(self, attr_key):
        return self.sess.run(self.batch_ops[attr_key])

    def __del__(self):
        print('{}: stop threads and close session!'.format(self.__class__.__name__))
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()


class STCSVLoader(object):
    def __init__(self, root, csv_path, attr_key, batch_size, height=331, width=331, transformer_fn=None,
                 num_epochs=None, shuffle=True, min_after_dequeue=25,
                 allow_smaller_final_batch=False, num_threads=2, seed=None):
        root = root.replace('\\', '/')
        root = root if root[-1] == '/' else '{}/'.format(root)
        self.root = root
        self.csv_path = csv_path.replace('\\', '/')
        self.attr_key = attr_key
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.transformer_fn = transformer_fn
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.min_after_dequeue = min_after_dequeue
        self.allow_smaller_final_batch = allow_smaller_final_batch
        self.num_threads = num_threads
        self.seed = seed

        df = pd.read_csv(self.csv_path, header=None)
        df = df[df[1].isin([attr_key])]

        if shuffle:
            df = df.sample(frac=1., random_state=seed)

        try:
            df[2] = df.apply(lambda line: line[2].index('y') + len(line[2]) if line[3] else line[2].index('y'), axis=1)
        except Exception as e:
            print(e)

        data_path = df[0].values
        label = df[2].values

        print('{}: create session!'.format(self.__class__.__name__))
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device('/cpu:0'):

                self.batch_ops, self.n_sample = csv_batch(root=self.root, data_path=data_path, label=label,
                                                          n_class=di.num_classes_v2[attr_key],
                                                          batch_size=batch_size,
                                                          height=height, width=width,
                                                          transformer_fn=transformer_fn, num_epochs=num_epochs,
                                                          shuffle=shuffle, min_after_dequeue=min_after_dequeue,
                                                          allow_smaller_final_batch=allow_smaller_final_batch,
                                                          num_threads=num_threads, seed=seed)
                if num_epochs is not None:
                    self.init = tf.local_variables_initializer()
        self.sess = session(graph=self.graph)
        if num_epochs is not None:
            self.sess.run(self.init)
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

    def __len__(self):
        return self.n_sample

    def batch(self):
        return self.sess.run(self.batch_ops)

    def __del__(self):
        print('{}: stop threads and close session!'.format(self.__class__.__name__))
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()


if __name__ == '__main__':
    from lib.datasets import transforms
    from experiment.hyperparams import HyperParams as hp

    transformer = transforms.Sequential(
        [
            transforms.Resize([331, 331]),
            transforms.SubtractMean([137.38, 131.21, 134.39]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]
    )
    # mt_loader = MTCSVLoader(root='E:/fashion-dataset/base',
    #                         csv_path='{}/dataset/labels/s1_label.csv'.format(hp.pro_path),
    #                         batch_size=16,
    #                         transformer_fn=transformer,
    #                         shuffle=True)
    st_loader = STCSVLoader(root='E:/fashion-dataset/base', attr_key='collar_design_labels',
                            csv_path='{}/dataset/labels/s1_label.csv'.format(hp.pro_path),
                            batch_size=16,
                            transformer_fn=transformer,
                            shuffle=True)
    # print(mt_loader.n_sample)
    print(st_loader.n_sample)

    for i in range(10):
        # batch1 = mt_loader.batch(attr_key='collar_design_labels')
        # batch2 = mt_loader.batch(attr_key='coat_length_labels')
        # print(batch1[0].shape, batch1[1].shape, list(map(lambda x: bytes.decode(x), batch1[2])))
        # print(batch2[0].shape)
        batch = st_loader.batch()
        print(batch[0].shape, batch[1].shape)
