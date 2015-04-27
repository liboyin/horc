__author__ = 'manabchetia'
# http://scipy-lectures.github.io/advanced/scikit-learn/

import pandas as pd
import numpy as np
import cv2
from os import listdir
from os.path import join, isfile
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cosine as cos_dist
from scipy.spatial.distance import euclidean as euclid_dist
import itertools as it
import pprint as pp
from rootsift import RootSIFT
import numpy as np



def get_img_files(img_dir):
    df = pd.DataFrame(filter(lambda x: ".JPG" in x, listdir(img_dir)), columns=["NAME"])
    df["CLASS"] = np.repeat(np.linspace(0, 49, num=50), 4) # number of files=50, number of classes=4
    df["CLASS"] = df["CLASS"].astype("category")
    return df


def extract_sift(df):
    key_points, kp_descriptors = [], []
    df_name = list(df["NAME"])
    # sift = cv2.SIFT(nfeatures=20)
    rs = RootSIFT()
    # Loop over each image
    for i in xrange(len(df)):
        img = cv2.imread(join(img_dir, df_name[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.FeatureDetector_create("SIFT")
        kps = detector.detect(img)
        # kp, desc = sift.detectAndCompute(img, None)
        (kps, descs) = rs.compute(img, kps)
        key_points.append(kps)
        # kp_descriptors.append(list(it.chain.from_iterable(desc)))
        kp_descriptors.append(descs.astype(np.uint16))
    df["SIFT_KP"] = key_points
    df["SIFT_KP_DESC"] = kp_descriptors
    return df


def prepare_data(df):
    df_desc = list(df["SIFT_KP_DESC"])
    df_class = list(df["CLASS"])
    X, y = [], []
    for img in xrange(len(df)):
        for dsc in df_desc[img]:
            X.append(dsc)
            y.append(df_class[img])
    return X, y

# def test(classifier, X):


if __name__ == '__main__':
    img_dir = '../data/uni/'
    df = get_img_files(img_dir)

    # df = extract_sift(df)
    # X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(df['NAME'], df['CLASS'], test_size=0.25, random_state=42)
    # X_train, y_train = prepare_data()

    # knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric=euclid_dist)
    # knn.fit(X_train, y_train)
    #
    # print(knn.score(X_test, y_test))

















