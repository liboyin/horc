"""
This script trains a kNN classifier based on HOG features
"""

__author__ = 'manabchetia'

from os import listdir
from os.path import join

import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from skimage import color, data
from skimage.feature import hog

from pyneural import pyneural

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
    df = pd.DataFrame(index=imgs, columns={'CLASS', 'HOG_DESC', 'TYPE'})
    df["CLASS"] = np.repeat(np.linspace(0, n_classes - 1, num=n_classes), n_files_per_class)
    return df


def extract_HOG(df):
    """
    This function extract GIST features from images and saves them in HOG_DESC column of dataframe
    :param df: dataframe containing filename as indices
    :return: dataframe containing HOG features
    """
    hog_desc = []
    # Loop over each image
    for img in list(df.index):
        img = data.imread(join(img_dir, img))
        img = color.rgb2gray(img)
        fd, _ = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
        hog_desc.append(fd.astype(np.float32))
    df['HOG_DESC'] = hog_desc
    return df


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
    img_dir = '../data/uni/'

    print('Reading image files ...')
    df = get_img_files(img_dir)

    print('Separating Training and Test files ...')
    X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(list(df.index), list(df['CLASS']),
                                                                            test_size=0.25, random_state=15)
    df.loc[X_test_file, 'TYPE'] = 'TEST'
    df.loc[X_train_file, 'TYPE'] = 'TRAIN'

    print('Extracting HOG features ...')
    df = extract_HOG(df)


    # # KNN
    # Get X, Y
    print('Getting X,Y for training ...')
    df_train = df[df['TYPE']=='TRAIN']

    X_train = list(df_train['HOG_DESC'])
    y_train = list(df_train['CLASS'])

    classifier = KNeighborsClassifier(n_neighbors=4, weights='distance', metric=cos_dist)
    classifier.fit(X_train, y_train)


    print('Testing ...')
    df_test = df[df['TYPE']=='TEST']

    X_test = list(df_test['HOG_DESC'])
    y_test = list(df_test['CLASS'])

    print('Accuracy: {}%'.format(classifier.score(X_test, y_test)*100))
