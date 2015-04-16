from __future__ import print_function
__author__ = 'manabchetia'

import cv2
import numpy as np
import pandas as pd
import random
from sklearn.svm import SVC

from os import listdir
from os.path import isfile, join
from pandas.io.pickle import read_pickle
from scipy.spatial.distance import cosine as cos_dist
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def init_df(img_dir):
    df = pd.DataFrame(filter(lambda x: ".JPG" in x, listdir(img_dir)), columns=["name"])
    df["class"] = np.repeat(np.linspace(0, 49, num=50), 4)
    df["class"] = df["class"].astype("category")
    key_points, kp_descriptors = list(), list()
    df_name = list(df["name"])
    sift = cv2.SIFT(nfeatures=20)
    for i in range(0, 200):
        progress = int(i / 200.0 * 1000) / 10.0
        print("{} {}%".format(df_name[i], progress), end="\r")
        img = cv2.imread(join(img_dir, df_name[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(img, None)
        key_points.append(map(lambda x: {"pt": x.pt, "angle": x.angle, "size": x.size,
                                         "octave": x.octave, "response": x.response}, kp))
        kp_descriptors.append(des.astype(np.uint16))
    df["sift_key_points"] = key_points
    df["sift_kp_descriptors"] = kp_descriptors
    return df

df_cache = "df.pickle"
if isfile(df_cache):
    df = read_pickle(df_cache)
else:
    df = init_df("../data/uni")
    df.to_pickle(df_cache)
# print(df.shape)
# print(df.dtypes)
# print(df.head())

def new_split():
    split = ["train"] * 200
    for i in range(0, 50):
        split[i * 4 + random.randint(0, 3)] = "test"
    return split
for i in range(0, 10):
    split_name = "split_" + str(i)
    df[split_name] = new_split()
    df[split_name] = df[split_name].astype("category")
# print(df.shape)
# print(df.head())

def get_training_set(split_id):
    df_train = df[df["split_" + str(split_id)] == "train"]
    df_train_sift_des = list(df_train["sift_kp_descriptors"])
    df_train_class = list(df_train["class"])
    train_x, train_y = list(), list()
    for i in range(0, 150):
        for d in df_train_sift_des[i]:
            train_x.append(d)
            train_y.append(df_train_class[i])
    return train_x, train_y

def test(classifier, split_id):
    df_test = df[df["split_" + str(split_id)] == "test"]
    df_test_sift_des = list(df_test["sift_kp_descriptors"])
    df_test_name = list(df_test["name"])
    predict = np.zeros(50, dtype=np.uint8)
    for i in range(0, 50):
        progress = int(i / 50.0 * 1000) / 10.0
        print("split_{} {} {}%".format(split_id, df_test_name[i], progress), end="\r")
        vote = np.zeros(50, dtype=np.uint16)
        for d in df_test_sift_des[i]:
            vote[classifier.predict(d)[0]] += 1
        predict[i] = vote.argmax()
    return predict

# classifier = KNeighborsClassifier(n_neighbors=5, weights="distance", metric=cos_dist)
classifier = SVC(C=1000000.0, gamma=0.0, kernel='rbf')
# scipy.spatial.distance.cosine is much faster than sklearn.metrics.pairwise.cosine_distances
accuracy = np.zeros(10, dtype=np.uint8)
for i in range(0, 10):
    train_x, train_y = get_training_set(i)
    predict = test(classifier.fit(train_x, train_y), i)
    accuracy[i] = sum(np.linspace(0, 49, num=50) == predict)
print(accuracy)
print("mean={}, sigma={}".format(np.mean(accuracy), np.std(accuracy)))