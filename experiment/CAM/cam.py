# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import tensorflow as tf


def grad_cam(logits, traget_conv):
    bt, nb_class = tf.shape(logits)[0], tf.shape(logits)[1]
    pred_class = tf.argmax(logits, axis=1)
    one_hot = tf.one_hot(pred_class, nb_class, on_value=1., off_value=0., axis=1)
    signal = tf.multiply(logits, one_hot)
    # print(signal) # Tensor("Mul:0", shape=(?, 8), dtype=float32)
    loss = tf.reduce_mean(signal, axis=1)
    grads = tf.gradients(loss, traget_conv)[0]
    # print(grads) # shape=(?, 8, 8, 1536)
    # Normalizing the gradients
    # [?, 8, 8, 1536]
    norm_grads = tf.divide(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))
    weights = tf.reduce_mean(norm_grads, axis=[1, 2])  # [?, 1536]
    weights = tf.reshape(weights, [bt, 1, 1, -1])

    # Taking a weighted average
    fuck = tf.multiply(weights, traget_conv)
    cam = tf.reduce_sum(fuck, axis=3) + tf.ones(tf.shape(traget_conv)[0: 3], dtype=tf.float32)

    # Passing through ReLU
    cam = tf.maximum(cam, 0)

    cam = cam / tf.reshape(tf.reduce_max(cam, axis=[1, 2]), [bt, 1, 1])

    return cam


def grad_cam_plus(logits, target_conv):
    # logits: [?, ?]
    # target_conv: [?, 8, 8, 1536]
    bt, nb_class = tf.shape(logits)[0], tf.shape(logits)[1]
    ch = tf.shape(target_conv)[3]
    pred_class = tf.cast(tf.argmax(logits, axis=1), tf.int32)

    # [?, ?]
    one_hot = tf.one_hot(pred_class, nb_class, on_value=1., off_value=0., axis=1)

    # [?, 8]
    signal = tf.multiply(logits, one_hot)

    # [?, 8, 8, 1536]
    target_grad = tf.gradients(signal, target_conv)[0]

    label = tf.expand_dims(pred_class, 1)
    index = tf.expand_dims(tf.range(0, bt), 1)
    indices = tf.concat([index, label], axis=1)
    fuck = tf.reshape(tf.gather_nd(tf.exp(signal), indices), [-1, 1, 1, 1])

    # first derivative
    # [?, 8, 8, 1536]
    first_grad = fuck * target_grad

    # second derivative
    second_grad = fuck * target_grad * target_grad

    # third derivative
    third_grad = fuck * target_grad * target_grad * target_grad

    global_sum = tf.reduce_sum(
        tf.reshape(target_conv, [bt, -1, ch]),
        axis=1
    )

    alpha_num = second_grad

    alpha_denom = second_grad * 2.0 + third_grad * tf.reshape(global_sum, [bt, 1, 1, ch])
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones(tf.shape(alpha_denom)))

    alphas = alpha_num / alpha_denom

    weights = tf.maximum(first_grad, 0.0)

    # [?, 1536]
    alpha_normalization_constant = tf.reduce_sum(tf.reduce_sum(alphas, axis=1), axis=1)
    alphas /= tf.reshape(alpha_normalization_constant, [bt, 1, 1, ch])

    deep_linear_weights = tf.reduce_sum(
        tf.reshape(weights * alphas, [bt, -1, ch]),
        axis=1
    )

    grad_cam_map = tf.reduce_sum(tf.reshape(deep_linear_weights, [bt, 1, 1, ch]) * target_conv, axis=3)

    # Passing through ReLU
    # [?, 8, 8]
    cam = tf.maximum(grad_cam_map, 0)
    cam = cam / tf.reshape(tf.reduce_max(cam, axis=[1, 2]), [bt, 1, 1])  # scale 0 to 1.0

    return cam


if __name__ == "__main__":
    import cv2
    import numpy as np
    from lib.networks.my_inceptionv4 import MyInceptionV4
    from lib.datasets.data_loader import MTCSVLoader
    from lib.datasets import transforms
    from experiment.hyperparams import HyperParams as hp
    from experiment.data_info import DataInfo as di
    from utils import session

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

    mt_loader = MTCSVLoader(root='E:/fashion-dataset/base',
                            csv_path='{}/dataset/labels/s1_label.csv'.format(hp.pro_path),
                            batch_size=hp.batch_size,
                            transformer_fn=transformer,
                            shuffle=False,
                            num_epochs=1,
                            allow_smaller_final_batch=True)
    sess = session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, '{}/model/MyInceptionV4/MultiTask/fashion-ai.ckpt'.format(hp.pro_path))
    for attr_key, n_class in di.num_classes_v2.items():
        print(attr_key)
        target_conv_layer = net.layers['PreGAP']
        logits = net.layers['cls_score_{}'.format(attr_key)]
        y_pred = tf.argmax(logits, axis=1)
        grad_cam_plus = grad_cam_plus(logits, target_conv_layer)
        grad_cam = grad_cam(logits, target_conv_layer)
        try:
            while not mt_loader.coord.should_stop():
                img_batch, label_batch, name_batch = mt_loader.batch(attr_key=attr_key)
                names = list(map(lambda v: bytes.decode(v), name_batch))
                cams, cam_pluses, yp, prob = sess.run(
                    [grad_cam, grad_cam_plus, y_pred, logits],
                    feed_dict={
                        net.data: img_batch,
                        net.is_training: False,
                        net.keep_prob: 1
                    })
                print(cams.shape)
                # 可视化CAM
                for idx, cam in enumerate(cams):
                    print(idx)
                    name = names[idx]
                    img = cv2.imread('E:/fashion-dataset/base/' + name, cv2.IMREAD_UNCHANGED)
                    cam = cv2.resize((cam * 255).astype(np.uint8), (img.shape[1], img.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
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
        except tf.errors.OutOfRangeError:
            print('Done.')
    sess.close()
