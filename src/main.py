from __future__ import print_function


import cv2
import numpy as np
import pandas as pd
import random
from os import listdir
from os.path import join
from scipy.spatial.distance import cosine

img_dir = "../data/uni"
df = pd.DataFrame(filter(lambda x: ".JPG" in x, listdir(img_dir)), columns=["name"])

df["class"] = np.repeat(np.linspace(0, 49, num=50), 4)
df["class"] = df["class"].astype("category")

split = ["train"] * 200
for i in range(0, 50):
    split[i * 4 + random.randint(0, 3)] = "test"
df["split"] = split
df["split"] = df["split"].astype("category")
print(df.dtypes)
print(df.head())

df_train = df[df["split"] == "train"]
df_train_name = list(df_train["name"])  # TODO: series to sequentially numpy array?
df_train_class = list(df_train["class"])
sift = cv2.SIFT(nfeatures=20)
support = list()
for i in range(0, 150):
    progress = int(i / 150.0 * 1000) / 10.0
    print("{}: {}%".format(df_train_name[i], progress), end="\r")
    img = cv2.imread(join(img_dir, df_train_name[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(img, None)
    for d in des:
        support.append((tuple(d), df_train_class[i]))

df_test = df[df["split"] == "test"]
df_test_name = list(df_test["name"])
sift = cv2.SIFT(nfeatures=10)
predict = list()
for i in range(0, 50):
    progress = int(i / 50.0 * 1000) / 10.0
    print("{}: {}%".format(df_train_name[i], progress), end="\r")
    img = cv2.imread(join(img_dir, df_test_name[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(img, None)
    vote = np.zeros(50, dtype=np.uint16)
    for d in des:
        top_supports = sorted(support, key=lambda (k, v): cosine(k, d))[0:5]
        for (k, v) in top_supports:
            vote[int(v)] += 1
    predict.append(vote.argmax())


correct = 0
for i in range(0, 50):
    if predict[i] == i:
        correct += 1
print(predict)
print(correct)
