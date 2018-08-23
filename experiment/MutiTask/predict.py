# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import multiprocessing
import tensorflow as tf
from lib.networks.my_inceptionv4 import MyInceptionV4
from lib.datasets.data_loader import MTCSVLoader
from lib.datasets import transforms
from experiment.hyperparams import HyperParams as hp
from experiment.data_info import DataInfo as di
from utils import session, csv_writer


def predict():
    data = tf.placeholder(tf.float32, [None, 331, 331, 3])
    label = tf.placeholder(tf.float32, [None, None])
    dropout = tf.placeholder(tf.float32)
    phase = tf.placeholder(tf.bool)
    net = MyInceptionV4(data, label, dropout, phase)

    # batch data
    transformer = transforms.Sequential(
        [
            transforms.Resize([331, 331]),
            transforms.Preprocess(),
            # transforms.RandomHorizontalFlip(),
        ]
    )

    mt_loader = MTCSVLoader(root='E:/fashion-dataset/rank',
                            csv_path='E:\\fashion-dataset\\rank\\Tests\\question.csv',
                            batch_size=64,
                            transformer_fn=transformer,
                            shuffle=False,
                            num_epochs=1,
                            allow_smaller_final_batch=True)
    sess = session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, '{}/model/MyInceptionV4/MultiTask/fashion-ai.ckpt'.format(hp.pro_path))
    queue = multiprocessing.Queue(maxsize=30)
    writer_process = multiprocessing.Process(target=csv_writer, args=['{}/result/sub.csv'.format(hp.pro_path), queue, 'stop'])
    writer_process.start()
    for attr_key, n_class in di.num_classes_v2.items():
        flat_logit = net.layers['cls_prob_{}'.format(attr_key)]
        y_pred = tf.argmax(flat_logit, axis=1)
        print('writing predictions...')
        try:
            while not mt_loader.coord.should_stop():
                img_batch, label_batch, name_batch = mt_loader.batch(attr_key=attr_key)
                names = list(map(lambda v: bytes.decode(v), name_batch))
                probs, preds = sess.run(
                    [flat_logit, y_pred],
                    feed_dict={
                        net.data: img_batch,
                        net.is_training: False,
                        net.keep_prob: 1
                    })
                queue.put(('continue', names, attr_key, probs))
                print(probs.shape)

        except tf.errors.OutOfRangeError:
            print('Predict {} Done.'.format(attr_key))
    queue.put(('stop', None, None, None))
    writer_process.join()
    sess.close()


if __name__ == '__main__':
    predict()
