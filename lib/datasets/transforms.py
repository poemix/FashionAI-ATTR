# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import cv2
import numbers
import collections
import numpy as np
import tensorflow as tf
from utils import apply_with_random_selector


class Sequential(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``image`` or ``numpy.ndarray`` to tensor."""

    def __init__(self, dtype=tf.float32):
        self.dtype = dtype

    def __call__(self, img):
        return tf.convert_to_tensor(img, dtype=self.dtype)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize the given image with mean and standard deviation."""

    def __init__(self, mean, std):
        assert (isinstance(mean, collections.Iterable))
        assert (isinstance(std, collections.Iterable))
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std
        self.size = len(mean)

    def __call__(self, img):
        mean = tf.constant(self.mean, dtype=tf.float32, shape=[1, 1, self.size], name='img_mean')
        std = tf.constant(self.std, dtype=tf.float32, shape=[1, 1, self.size], name='img_std')
        return tf.divide(tf.subtract(img, mean), std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomHorizontalFlip(object):
    """Horizontally flip the given image randomly with a probability of 0.5."""

    def __call__(self, img):
        re = tf.image.random_flip_left_right(img)
        return re

    def __repr__(self):
        return self.__class__.__name__ + '()'


class HorizontalFlip(object):
    """Horizontally flip the given image."""

    def __call__(self, img):
        re = tf.image.flip_left_right(img)
        return re

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomVerticalFlip(object):
    """Vertically flip the given image randomly with a probability of 0.5."""

    def __call__(self, img):
        return tf.image.random_flip_up_down(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class VerticalFlip(object):
    """Vertically flip the given image."""

    def __call__(self, img):
        return tf.image.flip_up_down(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Reshape(object):
    """Resize the input image to the given shape."""

    def __init__(self, shape):
        assert (isinstance(shape, collections.Iterable))
        self.shape = shape

    def __call__(self, img):
        return tf.reshape(img, tf.stack(self.shape))

    def __repr__(self):
        return self.__class__.__name__ + '(shape={0})'.format(self.shape)


class Resize(object):
    """Resize the input image to the given size."""

    def __init__(self, size, interpolation=0):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return tf.image.resize_images(img, self.size, method=self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def distort_color(image, color_ordering=0):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        elif color_ordering == 2:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_brightness(image, max_delta=32.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32.)
        else:
            raise ValueError('color_ordering must be in [0, 3]')
        image = tf.clip_by_value(image, 0., 255.)
        return image

    def __call__(self, img):
        img = apply_with_random_selector(
            img,
            lambda x, ordering: self.distort_color(x, ordering),
            num_cases=4
        )
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomRotation(object):
    """Random rotate the given image by angle."""

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    @staticmethod
    def random_rotate(image, low, high):
        def py2tf_func(im, lo, hi):
            h, w, _ = im.shape
            # 旋转角度范围
            angle = np.random.uniform(low=lo, high=hi + 1)
            # print(angle)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle=angle, scale=1)
            dst = cv2.warpAffine(im, M, dsize=(w, h), borderValue=(255., 255., 255.))
            return dst

        ny, nx, _ = image.shape
        image = tf.py_func(py2tf_func, [image, low, high], tf.float32)
        image.set_shape([ny, nx, 3])
        return image

    def __call__(self, img):
        return self.random_rotate(img, self.degrees[0], self.degrees[1])

    def __repr__(self):
        return self.__class__.__name__ + '(degrees={0})'.format(self.degrees)


class SubtractMean(object):
    """Zero-Center the given image."""

    def __init__(self, mean):
        assert (isinstance(mean, collections.Iterable))
        self.mean = mean
        self.size = len(mean)

    def __call__(self, img):
        re = tf.subtract(img, tf.constant(self.mean, dtype=tf.float32, shape=[1, 1, self.size], name='img_mean'))
        return re

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0})'.format(self.mean)


class Preprocess(object):
    """Preprocess."""

    def __call__(self, img):
        re = tf.subtract(tf.divide(img, 127.5), 1)
        return re

    def __repr__(self):
        return self.__class__.__name__ + '()'
