# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from experiment.hyperparams import HyperParams as hp
from experiment.data_info import DataInfo as di
from utils import preprocess_label


def split_dataset(phase='train'):
    path = '{}/dataset/labels/{}'
    s1_path = path.format(hp.pro_path, 's1_label.csv')
    s1ab_path = path.format(hp.pro_path, 's1ab_label.csv')
    s2_path = path.format(hp.pro_path, 's2_label.csv')

    s1_df = pd.read_csv(s1_path, header=None)
    s1ab_df = pd.read_csv(s1ab_path, header=None)
    s2_df = pd.read_csv(s2_path, header=None)

    s2_df = s2_df.sample(frac=1, random_state=hp.seed)

    datas = []
    labels = []
    attr_keys = []
    singles = []

    for attr_key, n_class in di.num_classes.items():
        s1_data = s1_df[s1_df[1].isin([attr_key])][0].values
        s1_label = s1_df[s1_df[1].isin([attr_key])][2].values
        s1_single = s1_df[s1_df[1].isin([attr_key])][3].values

        s1ab_data = s1ab_df[s1ab_df[1].isin([attr_key])][0].values
        s1ab_label = s1ab_df[s1ab_df[1].isin([attr_key])][2].values
        s1ab_single = s1ab_df[s1ab_df[1].isin([attr_key])][3].values

        s2_data = s2_df[s2_df[1].isin([attr_key])][0].values
        s2_label = s2_df[s2_df[1].isin([attr_key])][2].values
        s2_single = s2_df[s2_df[1].isin([attr_key])][3].values

        # 对复赛数据进行9:1分层抽样(按类别数)
        s2_y = preprocess_label(s2_label, len(s2_label), n_class)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=0.9, test_size=0.1, random_state=hp.seed)

        train_indices = []
        eval_indices = []

        for train_idx, eval_idx in sss.split(s2_data, s2_y):
            train_indices.append(train_idx)
            eval_indices.append(eval_idx)

        train_idx = train_indices[0]
        eval_idx = eval_indices[0]

        train_data = s2_data[train_idx]
        eval_data = s2_data[eval_idx]
        train_label = s2_label[train_idx]
        eval_label = s2_label[eval_idx]
        train_single = s2_single[train_idx]
        eval_single = s2_single[eval_idx]

        # 初赛数据
        for path in s1_data:
            datas.append(path)
            attr_keys.append(attr_key)
        for v in s1_label:
            labels.append(v)
        for s in s1_single:
            singles.append(s)

        # 初赛答案数据
        for path in s1ab_data:
            datas.append(path)
            attr_keys.append(attr_key)
        for v in s1ab_label:
            labels.append(v)
        for s in s1ab_single:
            singles.append(s)

        if phase == 'train':
            sdata = train_data
            slabel = train_label
            ssingle = train_single
        elif phase == 'eval':
            sdata = eval_data
            slabel = eval_label
            ssingle = eval_single
        else:
            raise ValueError('error.')

        # 复赛数据
        for path in sdata:
            datas.append(path)
            attr_keys.append(attr_key)
        for v in slabel:
            labels.append(v)
        for s in ssingle:
            singles.append(s)

    df = pd.DataFrame()
    df['name'] = datas
    df['attr_key'] = attr_keys
    df['label'] = labels
    df['is_single'] = singles
    df.to_csv('{}.csv'.format(phase), index=False, header=None)


if __name__ == '__main__':
    # split_dataset('train')
    split_dataset('eval')
