# -*- coding: utf-8 -*-

# @Env      : windows python3.5 tensorflow1.4.0
# @Author   : xushiqi
# @Email    : xushiqitc@163.com
# @Software : PyCharm


import pandas as pd


def simple_ensemble(save_name='sub_ensemble.csv'):
    df1 = pd.read_csv('sub1.csv', header=None)
    df2 = pd.read_csv('sub2.csv', header=None)

    names = []
    aks = []
    new_probs = []

    for raw in df1.values:
        names.append(raw[0])
        aks.append(raw[1])

        prob = raw[2].split(';')
        prob2 = df2[df2[0].isin([raw[0]])].values[0][2].split(';')
        tmp_result = ''
        for idx, v in enumerate(prob):
            tmp_result += '{:.4f};'.format(0.5*float(v) + 0.5*float(prob2[idx]))
        new_probs.append(tmp_result[:-1])

    sub_df = pd.DataFrame()
    sub_df['name'] = names
    sub_df['ak'] = aks
    sub_df['result'] = new_probs
    sub_df.to_csv(save_name, index=False, header=None)


if __name__ == '__main__':
    simple_ensemble()
