# Python 2
from __future__ import print_function

import numpy as np

from data import num_sift_kp


class SetNearestNeighbourClassifier:
    # C: # of classes; N: # of data point per training class; D: dimension of data point
    def __init__(self, dist_kernel):  # (R^(A*D), R^(B*D) -> R^(A*B))
        self.dist_kernel = dist_kernel

    def fit(self, train_x):  # C * N * D
        self.train_x = train_x

    def predict(self, test_x):  # 1 * D
        return np.array([np.min(self.dist_kernel([test_x], x)) for x in self.train_x])  # 1 * C
        # the closest distance, according to @dist_kernel, form @test_x to any vector in all classes of @train_x


def get_training_set(df, split_id):
    df_train = df[df["split_" + str(split_id)] == "train"]
    sift_des_list = list(df_train["sift_kp_descriptors"])
    return np.array([np.vstack(sift_des_list[j*3:j*3+3]) for j in range(0, 50)])  # 50 * (3 * @num_sift_kp) * 128


def test(df, split_id, classifier):
    df_test = df[df["split_" + str(split_id)] == "test"]
    test_x = np.array(list(df_test["sift_kp_descriptors"]))  # 50 * @num_sift_kp * 128
    predict_y = np.empty(50, dtype=np.uint8)
    for i in range(0, 50):
        sum_dists = np.zeros(50, dtype=np.float32)
        for j in range(0, num_sift_kp):
            progress = int((i * num_sift_kp + j) / (50.0 * num_sift_kp) * 1000) / 10
            print("split_{} {}%".format(split_id, progress), end="\r")
            sum_dists += classifier.predict(test_x[i][j])
        predict_y[i] = np.argmin(sum_dists)
    return predict_y