"""
This script trains a kNN classifier based on SIFT features
"""

__author__ = 'manabchetia'

from os import listdir
from os.path import join

import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean

img_dir = '../data/uni/'
n_classes = 50
n_files_per_class = 4


def get_img_files(img_dir):
    """
    This function reads the filenames from a directory, assigns labels to the files and saves them to a dataframe
    :param img_dir: path of images
    :return: dataframe
    """
    imgs = filter(lambda x: ".JPG" in x, listdir(img_dir))
    df = pd.DataFrame(index=imgs, columns={'CLASS', 'SIFT_KP', 'SIFT_KP_DESC', 'TYPE'})
    df["CLASS"] = np.repeat(np.linspace(0, n_classes - 1, num=n_classes), n_files_per_class)
    return df


def extract_SIFT(df):
    """
    This function extract SIFT keypoints and descriptors from images and saves them in SIFT_KP, SIFT_KP_DESC column of dataframe respectively
    :param df: dataframe containing filename as indices
    :return: dataframe containing SFIT keypoints and descriptors
    """
    key_points, kp_descriptors = [], []

    sift = cv2.SIFT(nfeatures=25)

    # Loop over each image
    for img in list(df.index):
        img = cv2.imread(join(img_dir, img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, descs = sift.detectAndCompute(img, None)
        kp_descriptors.append(descs.astype(np.float32))
    df['SIFT_KP'] = key_points
    df['SIFT_KP_DESC'] = kp_descriptors
    return df


def get_X_Y(df):
    """
    This function gets features (X) and labels (Y) from a dataframe
    :param df: keypoints and descriptors
    :return: lists containing features (X), labels (Y)
    """
    X, y = [], []
    for img in list(df.index):
        for dsc in df.loc[img, 'SIFT_KP_DESC']:
            X.append(dsc)
            y.append(df.loc[img, 'CLASS'])
    return X, y


def get_predictions(df, classifier):
    """
    This function predicts the output of a classifier
    :param df: data frame containing SIFT keypoints and descriptors
    :param classifier: any classifier such as kNN, SVM from scikit-learn library
    :return: predictions from learnt classifier
    """
    predictions = []
    for img in list(df.index):
        votes = {i: 0 for i in xrange(n_classes)}
        # Get majority votes
        for dsc in df.loc[img, 'SIFT_KP_DESC']:
            votes[classifier.predict(dsc)[0]] += 1

        predictions.append(max(votes, key=votes.get))

    return predictions


def get_accuracy(predictions, truth):
    """
    This function prints the accuracy
    :param predictions: predicted lables
    :param truth: ground thruth
    :return: accuracy in %
    """
    correct = sum(1 for p, t in zip(predictions, truth) if p == t)
    return correct * 100 / len(predictions)


if __name__ == '__main__':
    print('Reading image files ...')
    df = get_img_files(img_dir)

    print('Separating Training and Test files ...')
    X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(list(df.index), list(df['CLASS']),
                                                                            test_size=0.25, random_state=42)
    df.loc[X_test_file, 'TYPE'] = 'TEST'
    df.loc[X_train_file, 'TYPE'] = 'TRAIN'

    print('Extracting SIFT features ...')
    df = extract_SIFT(df)

    # KNN
    # Get X, Y
    print('Getting X,Y for training ...')
    df_train = df[df['TYPE'] == 'TRAIN']
    X_train_dsc, y_train_dsc = get_X_Y(df[df['TYPE'] == 'TRAIN'])

    print('Training Classifier ...')
    classifier = KNeighborsClassifier(n_neighbors=4, weights='distance', metric=euclidean)
    classifier.fit(X_train_dsc, y_train_dsc)

    print('Testing ...')
    df_test = df[df['TYPE'] == 'TEST']
    predictions = get_predictions(df_test, classifier)

    print('Accuracy: {} %'.format(get_accuracy(predictions, list(df_test['CLASS']))))
