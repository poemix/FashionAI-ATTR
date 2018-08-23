# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


class DataInfo(object):
    num_classes = dict(coat_length_labels=8, collar_design_labels=5,
                       lapel_design_labels=5, neck_design_labels=5,
                       neckline_design_labels=10, pant_length_labels=6,
                       skirt_length_labels=6, sleeve_length_labels=9)

    num_classes_v2 = dict(coat_length_labels=8 * 2, collar_design_labels=5,
                          lapel_design_labels=5, neck_design_labels=5,
                          neckline_design_labels=10, pant_length_labels=6 * 2,
                          skirt_length_labels=6 * 2, sleeve_length_labels=9 * 2)
