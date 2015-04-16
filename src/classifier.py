__author__ = 'manabchetia'

import pandas as pd
# from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like, repeat, linspace, array
import numpy as np
import cv2
from os import listdir
from os.path import join, isfile
from sklearn.cross_validation import train_test_split
from scipy.spatial.distance import cosine
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pprint as pp
from scipy.spatial.distance import cosine as cos_dist

PRE_ALLOCATION_BUFFER = 1000

def get_img_files(img_dir):
    df = pd.DataFrame(filter(lambda x: ".JPG" in x, listdir(img_dir)), columns=["name"])
    df["class"] = np.repeat(np.linspace(0, 49, num=50), 4) # number of files=50, number of classes=4
    df["class"] = df["class"].astype("category")
    return df

def extract_sift(df):
    key_points, kp_descriptors = [], []
    df_name = list(df["name"])
    sift = cv2.SIFT(nfeatures=20)
    for i in xrange(0, 200):
        img = cv2.imread(join(img_dir, df_name[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(img, None)
        key_points.append(map(lambda x: {"pt": x.pt, "angle": x.angle, "size": x.size,
                                         "octave": x.octave, "response": x.response}, kp))
        kp_descriptors.append(des.astype(np.uint16))
    df["sift_kp"] = key_points
    df["sift_kp_desc"] = kp_descriptors
    return df


if __name__ == '__main__':
    img_dir = '../data/uni/'
    df = get_img_files(img_dir)
    df = extract_sift(df)
    X_train, X_test, y_train, y_test = train_test_split(df["sift_kp_desc"], df["class"], test_size=0.25, random_state=42)


    knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric=cos_dist)
    print knn.score(X_test, y_test)













