import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import math


def is_feature_discrete(data, feature):
    return data[feature].nunique() < 10


def entropy_feature_value(data, feature, feature_value, targ):
    filtered_data = data[data[feature] == feature_value]
    entropy = 0.0
    for count in filtered_data[targ].value_counts().array:
        p = count / len(filtered_data)
        if p != 0.0:
            entropy -= p * math.log2(p)
    return entropy


def target_entropy(data, targ):
    h = 0.0
    for count in data[targ].value_counts().array:
        p = count / len(data[targ])
        if p != 0.0:
            h -= p * math.log2(p)
    return h


def information_gain(data, feature, targ):
    res = target_entropy(data, targ)
    for feat, c in data[feature].value_counts().items():
        p = c / len(data)
        res -= p * entropy_feature_value(data, feature, feat, targ)
    return res


def intrinstic_information(data, feature):
    r = 0.0
    for count in data[feature].value_counts().array:
        p = count / len(data)
        r -= p * math.log2(p)
    return r


def gain_ratio(data, feature, targ):
    return information_gain(data, feature, targ) / intrinstic_information(data, feature)


def clear_dataset(data, targ):
    for col in data.columns:
        miss = np.mean(data[col].isnull())
        if miss > 0.2:
            data.drop(col, axis=1, inplace=True)
        # m = data[col].median()
        # data[col] = data[col].fillna(m)
    # data.boxplot(column=['Processor_Speed'])
    # print(data['Processor_Speed'].describe())
    # data['Storage_Capacity'].value_counts().plot.bar()
    num_rows = len(data.index)
    # too much same values
    no_inf_features = []
    for col in data.columns:
        counts = data[col].value_counts(dropna=False)
        pct_same = (counts / num_rows).iloc[0]
        if pct_same > 0.8:
            no_inf_features.append(col)
    for col in no_inf_features:
        data.drop(col, axis=1, inplace=True)
    target_corr = data.corrwith(data[targ])
    good_features = target_corr[abs(target_corr.abs()) > 0.03].index.tolist()
    return data[good_features]


if __name__ == '__main__':
    laptops = pd.read_csv('Laptop_price.csv')
    target = 'Price'
    laptops['Brand'], _ = pd.factorize(laptops['Brand'])
    laptops = clear_dataset(laptops, target)
    corr_matrix = laptops.corr()
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    # plt.show()
    laptops[target] = pd.cut(laptops[target], bins=3)

    for f in laptops.columns.array:
        if not is_feature_discrete(laptops, f):
            laptops[f] = pd.cut(laptops[f], bins=3)
        if f != target:
            print(f + ' gain ratio', gain_ratio(laptops, f, target))
