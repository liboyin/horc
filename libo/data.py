# Python 2
from __future__ import print_function

import cv2
import numpy as np
import pandas as pd
import pickle

from os import listdir
from os.path import isfile, join

img_dir = "data"
num_sift_kp = 25


def get_sift_descriptors(sift, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, des = sift.detectAndCompute(img, None)
    if len(des) > num_sift_kp:
        des = des[:num_sift_kp]
    return des.astype(np.uint32)


def get_hue_histogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue_vec = img[:, :, 0].ravel()
    return np.histogram(hue_vec, bins=181, range=(0, 180))[0]


def make_df():
    df = pd.DataFrame(filter(lambda x: ".JPG" in x, listdir(img_dir)), columns=["name"])
    df["class"] = np.repeat(np.arange(50), 4)
    df["class"] = df["class"].astype("category")
    sift_kp_des, hue_hist = list(), list()
    df_name = list(df["name"])
    sift = cv2.SIFT(nfeatures=num_sift_kp)  # @nfeatures is only approximate
    for i in range(0, 200):
        progress = int(i / 200.0 * 1000) / 10
        print("{} {}%".format(df_name[i], progress), end="\r")
        img = cv2.imread(join(img_dir, df_name[i]))
        sift_kp_des.append(get_sift_descriptors(sift, img))
        hue_hist.append(get_hue_histogram(img))
    df["sift_kp_descriptors"] = sift_kp_des
    df["hue_histogram"] = hue_hist
    return df


def get_df():
    df_cache = "df_{}.pickle".format(num_sift_kp)
    if isfile(df_cache):
        with open(df_cache, mode="rb") as h:
            df = pickle.load(h)
    else:
        df = make_df()
        with open(df_cache, mode="wb") as h:
            pickle.dump(df, h, protocol=2)
    return df