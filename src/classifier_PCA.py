"""
This script does PCA on raw pixels and runs kNN for classification
"""

__author__ = 'manabchetia'

from os import listdir
from os.path import join

import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean


# GLOBAL Variables
img_dir = '../data/uni/' # Set Path for Images
n_classes = 50 # Set Number of classes
n_files_per_class = 4 # Set Number of files per class


def get_img_files(img_dir):
    """
    This function reads the filenames from a directory, assigns labels to the files and saves them to a dataframe
    :param img_dir: path of images
    :return: dataframe
    """
    imgs = filter(lambda x: ".JPG" in x, listdir(img_dir))
    df = pd.DataFrame(index=imgs, columns={'CLASS', 'PIX_DESC', 'TYPE'})
    df["CLASS"] = np.repeat(np.linspace(0, n_classes - 1, num=n_classes), n_files_per_class)
    return df


def extract_Pixels(df):
    """
    This function reads each images and saves them in PIX_DESC column of dataframe
    :param df: dataframe containing filename as indices
    :return: dataframe containing raw pixel values
    """
    pix_desc = []
    # Loop over each image
    for img in list(df.index):
        img = Image.open(join(img_dir, img))
        img = np.asarray(list(img.getdata()))
        rows, cols = img.shape
        img_flat = img.reshape(1, rows * cols)
        pix_desc.append(img_flat[0])
    df['PIX_DESC'] = pix_desc
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
    print('Reading image files ...')
    df = get_img_files(img_dir)

    print('Separating Training and Test files ...')
    X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(list(df.index), list(df['CLASS']),
                                                                            test_size=0.25, random_state=15)
    df.loc[X_test_file, 'TYPE'] = 'TEST'
    df.loc[X_train_file, 'TYPE'] = 'TRAIN'

    print('Extracting RAW pixel values ...')
    df = extract_Pixels(df)


    # Get X, Y
    print('Getting X,Y for training ...')
    df_train = df[df['TYPE'] == 'TRAIN']

    X_train = list(df_train['PIX_DESC'])
    y_train = list(df_train['CLASS'])

    print 'Extracting Principal Components'
    pca = PCA(n_components=5)
    X_train = pca.fit_transform(X_train)

    # KNN
    print 'Training'
    classifier = KNeighborsClassifier(n_neighbors=4, weights='distance', metric=euclidean)
    classifier.fit(X_train, y_train)

    print('Testing ...')
    df_test = df[df['TYPE'] == 'TEST']

    X_test = list(df_test['PIX_DESC'])
    X_test = pca.transform(X_test)
    y_test = list(df_test['CLASS'])

    print('Accuracy: {}%'.format(classifier.score(X_test, y_test) * 100))
