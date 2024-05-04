import calendar
import random

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


if __name__ == '__main__':
    laptops = pd.read_csv('Laptop_price.csv')
    target = 'Price'
    laptops[target] = pd.cut(laptops[target], bins=3)
    for f in laptops.columns.array:
        if not is_feature_discrete(laptops, f):
            laptops[f] = pd.cut(laptops[f], bins=3)
        if f != target:
            print(f, gain_ratio(laptops, f, target))
