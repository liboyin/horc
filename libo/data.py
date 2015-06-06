from __future__ import print_function  # Python 2

import cv2
import numpy as np
import pandas as pd
import pickle
import random

from os import listdir
from os.path import isfile, join

img_dir = 'data'
num_sift_kp = 30
C = 50  # C: # of classes
M = 4  # M: # of images per class


def get_sift_descriptors(sift, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, des = sift.detectAndCompute(img, None)
    if len(des) > num_sift_kp:
        des = des[:num_sift_kp]
    return des.astype(np.uint32)


def get_hsv_histograms(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue_hist = np.histogram(img[:, :, 0].ravel(), bins=181, range=(0, 180))[0]  # [0, 180] closed interval
    sat_hist = np.histogram(img[:, :, 1].ravel(), bins=256, range=(0, 255))[0]  # [0, 255] closed interval
    val_hist = np.histogram(img[:, :, 2].ravel(), bins=256, range=(0, 255))[0]
    return hue_hist, sat_hist, val_hist


def get_rgb_histograms(img):
    red_hist = np.histogram(img[:, :, 0].ravel(), bins=256, range=(0, 255))[0]
    green_hist = np.histogram(img[:, :, 1].ravel(), bins=256, range=(0, 255))[0]
    blue_hist = np.histogram(img[:, :, 2].ravel(), bins=256, range=(0, 255))[0]
    return red_hist, green_hist, blue_hist


def make_df():
    df = pd.DataFrame(filter(lambda x: '.JPG' in x, listdir(img_dir)), columns=['name'])
    df['class'] = np.repeat(np.arange(C), M)
    df['class'] = df['class'].astype('category')
    print('loading images...', end='\r')
    images = [cv2.imread(join(img_dir, x)) for x in df['name']]
    print('extracting SIFT descriptors...', end='\r')
    sift = cv2.SIFT(nfeatures=num_sift_kp)  # @nfeatures is only approximate
    sift_kp_des = [get_sift_descriptors(sift, x) for x in images]
    df['sift_kp_descriptors'] = sift_kp_des
    print('extracting RGB histograms...', end='\r')
    rgb_hist = [get_rgb_histograms(x) for x in images]
    df['red_histogram'], df['green_histogram'], df['blue_histogram'] = list(zip(*rgb_hist))
    print('extracting HSV histograms...', end='\r')
    hsv_hist = [get_hsv_histograms(x) for x in images]
    df['hue_histogram'], df['saturation_histogram'], df['value_histogram'] = list(zip(*hsv_hist))
    return df


def lazy_df():
    df_cache = 'df_{}.pickle'.format(num_sift_kp)
    if isfile(df_cache):
        with open(df_cache, mode='rb') as h:
            return pickle.load(h)
    df = make_df()
    with open(df_cache, mode='wb') as h:
        pickle.dump(df, h, protocol=2)
    return df


def new_split():
    split = ['train'] * C * M
    for i in range(0, C):
        split[i * M + random.randint(0, M - 1)] = 'test'  # one datapoint in each class is used as test
    return split


def get_split_df(num_split):
    df = lazy_df()
    print('splitting train/test...', end='\r')
    for i in range(0, num_split):
        split_name = 'split_' + str(i)
        df[split_name] = new_split()
        df[split_name] = df[split_name].astype('category')
    return df


def get_train_x(df, split_id, feature, transform):  # transform: R^{3*D} -> ?
    df_train_feature = df[df['split_' + str(split_id)] == 'train'][feature].reset_index(drop=True)
    return np.array([transform(df_train_feature[i*(M-1):(i+1)*(M-1)]) for i in range(0, C)])


def get_test_x(df, split_id, feature):
    return np.array(list(df[df['split_' + str(split_id)] == 'test'][feature]))
    # convert to list then to numpy array in case each cell in @df[feature] has dimension higher than 1
