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


def make_df():
    df = pd.DataFrame(filter(lambda x: ".JPG" in x, listdir(img_dir)), columns=["name"])
    df["class"] = np.repeat(np.arange(50), 4)
    df["class"] = df["class"].astype("category")
    key_points, kp_descriptors = list(), list()
    df_name = list(df["name"])
    sift = cv2.SIFT(nfeatures=num_sift_kp)  # @nfeatures is only approximate
    for i in range(0, 200):
        progress = int(i / 200.0 * 1000) / 10.0
        print("{} {}%".format(df_name[i], progress), end="\r")
        img = cv2.imread(join(img_dir, df_name[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(img, None)
        if len(kp) > num_sift_kp:
            kp = kp[:num_sift_kp]
            des = des[:num_sift_kp]
        key_points.append(map(lambda x: {"pt": x.pt, "angle": x.angle, "size": x.size,
                                         "octave": x.octave, "response": x.response}, kp))
        kp_descriptors.append(des.astype(np.uint16))
    df["sift_key_points"] = key_points
    df["sift_kp_descriptors"] = kp_descriptors
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