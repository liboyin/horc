__author__ = 'manabchetia'

import pandas as pd
from os import listdir
from os.path import join
from pyneural import pyneural
from sklearn.cross_validation import train_test_split
import numpy as np
import cv2
from skimage import color, data
from skimage.feature import hog
from rootsift import RootSIFT
from scipy.spatial.distance import euclidean as euclid_dist
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from scipy.spatial.distance import cosine as cos_dist

img_dir = '../data/uni/'
n_classes = 50
n_files_per_class = 4

def get_img_files(img_dir):
    imgs = filter(lambda x: ".JPG" in x, listdir(img_dir))
    df = pd.DataFrame(index=imgs, columns={'CLASS', 'HOG_DESC', 'TYPE'})
    df["CLASS"] = np.repeat(np.linspace(0, n_classes-1, num=n_classes), n_files_per_class)
    return df


def extract_HOG(df):
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
    correct = sum(1 for p, t in zip(predictions, truth) if p == t)
    return correct * 100 / len(predictions)


if __name__ == '__main__':
    img_dir = '../data/uni/'

    print('Reading image files ...')
    df = get_img_files(img_dir)

    print('Separating Training and Test files ...')
    X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(list(df.index), list(df['CLASS']), test_size=0.25, random_state=15)
    df.loc[X_test_file,  'TYPE'] = 'TEST'
    df.loc[X_train_file, 'TYPE'] = 'TRAIN'

    print('Extracting HOG features ...')
    df = extract_HOG(df)

    # NEURAL NETWORK
    # Get X, Y
    print('Getting X,Y for training ...')
    df_train = df[df['TYPE'] == 'TRAIN']

    features_train = np.asarray(list(df_train['HOG_DESC']))
    labels_train = np.asarray(list(df_train['CLASS']), dtype=np.int8)

    n_rows, n_features = features_train.shape
    n_labels = 50

    labels_expanded = np.zeros((n_rows, n_labels), dtype=np.int8)
    for i in xrange(n_rows):
        labels_expanded[i][labels_train[i]] = 1

    print('Training ...')
    nn = pyneural.NeuralNet([n_features, (n_features + n_labels)/2,  n_labels])
    nn.train(features_train, labels_expanded, 500, 40, 0.005, 0.0,
             1.0)  # features, labels, iterations, batch size, learning rate, L2 penalty, decay multiplier

    print('Testing ...')
    df_test = df[df['TYPE'] == 'TEST']

    features_test = np.asarray(list(df_test['HOG_DESC']))
    labels_test = np.asarray(list(df_test['CLASS']))

    predictions = np.asarray(nn.predict_label(features_test), dtype=np.int8)
    print(predictions)
    print(" ")
    print(labels_test)
    print('Accuracy: {} %'.format(get_accuracy(predictions, labels_test)))


    # KNN
    # # Get X, Y
    # print('Getting X,Y for training ...')
    # df_train = df[df['TYPE']=='TRAIN']
    #
    # X_train = list(df_train['HOG_DESC'])
    # y_train = list(df_train['CLASS'])
    #
    # classifier = KNeighborsClassifier(n_neighbors=4, weights='distance', metric=cos_dist)
    # classifier.fit(X_train, y_train)
    #
    #
    # print('Testing ...')
    # df_test = df[df['TYPE']=='TEST']
    #
    # X_test = list(df_test['HOG_DESC'])
    # y_test = list(df_test['CLASS'])
    #
    # print('Accuracy: {}%'.format(classifier.score(X_test, y_test)*100))




