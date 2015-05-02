from __future__ import print_function  # Python 2

import numpy as np

from data import num_sift_kp


class SoftNearestClassClassifier:  # a class may contain several clusters of vector
    # C: # of classes; N: # of vectors per training class; D: dimension of a vector
    def __init__(self, cdist):
        self.cdist = cdist  # collection pairwise distance: R^(A*D), R^(B*D) -> R^(A*B)

    def fit(self, train_x):
        self.train_x = train_x  # C * N * D

    def predict(self, test_x):  # 1 * D
        dists = np.array([np.min(self.cdist([test_x], x)) for x in self.train_x])  # 1 * C
        # closest Euclidean distance form @test_x to any vector in all classes of @train_x
        probs = dists.max() - dists  # unnormalized probability defined as the difference from the maximum distance
        return np.divide(probs, probs.sum())


class SoftNearestNeighbourClassifier:
    # C: # of classes; D: dimension of data point
    def __init__(self, psim):
        self.psim = psim  # pairwise similarity: R^D, R^D -> R

    def fit(self, train_x):
        self.train_x = train_x  # C * D

    def predict(self, test_x):  # 1 * D
        sims = np.array([self.psim(test_x, x) for x in self.train_x])  # 1 * C
        return np.true_divide(sims, sims.sum())  # normalized probability defined as the ratio to summed similarities


def get_train_sift(df, split_id):
    df_train = df[df["split_" + str(split_id)] == "train"]
    sift_des_list = list(df_train["sift_kp_descriptors"])
    return np.array([np.vstack(sift_des_list[i*3:i*3+3]) for i in range(0, 50)])  # 50 * (3 * @num_sift_kp) * 128


def get_train_hue(df, split_id):
    df_train = df[df["split_" + str(split_id)] == "train"]
    hue_hist_list = list(df_train["hue_histogram"])
    return np.array([np.sum(hue_hist_list[i*3:i*3+3], axis=0)/3 for i in range(0, 50)])  # 50 * @num_hue_bins


def test_sift_hue(df, split_id, sift_classifier, hue_classifier):
    df_test = df[df["split_" + str(split_id)] == "test"]
    test_x_sift = np.array(list(df_test["sift_kp_descriptors"]))  # 50 * @num_sift_kp * 128
    test_x_hue = np.array(list(df_test["hue_histogram"]))  # 50 * @num_hue_bins
    predict_y = np.empty(50, dtype=np.uint8)
    for i in range(0, 50):
        log_probs = np.zeros(50, dtype=np.float32)
        for j in range(0, num_sift_kp):
            progress = int((i * num_sift_kp + j) / (50.0 * num_sift_kp) * 1000) / 10
            print("split_{} {}%".format(split_id, progress), end="\r")
            log_probs += np.log(0.0000001 + sift_classifier.predict(test_x_sift[i][j]))
        log_probs += 0.5 * num_sift_kp * np.log(hue_classifier.predict(test_x_hue[i]))
        # print("class {}:\n{}".format(i, log_probs))
        predict_y[i] = np.argmax(log_probs)
    return predict_y