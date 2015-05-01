# Python 2
from __future__ import print_function

import numpy as np


def get_training_set(df, split_id):
    df_train = df[df["split_" + str(split_id)] == "train"]
    sift_des_list = list(df_train["sift_kp_descriptors"])
    class_list = list(df_train["class"])
    train_x, train_y = list(), list()
    for i in range(0, 150):
        for d in sift_des_list[i]:
            train_x.append(d)
            train_y.append(class_list[i])
    return train_x, train_y


def test(df,  split_id, classifier):
    df_test = df[df["split_" + str(split_id)] == "test"]
    sift_des_list = list(df_test["sift_kp_descriptors"])
    name_list = list(df_test["name"])
    predict_y = np.empty(50, dtype=np.uint8)
    for i in range(0, 50):
        progress = int(i / 50.0 * 1000) / 10
        print("split_{} {} {}%".format(split_id, name_list[i], progress), end="\r")
        vote = np.zeros(50, dtype=np.uint16)
        for d in sift_des_list[i]:
            vote[classifier.predict(d)[0]] += 1
        predict_y[i] = vote.argmax()
    return predict_y