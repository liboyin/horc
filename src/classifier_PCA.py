__author__ = 'manabchetia'

__author__ = 'manabchetia'

import pandas as pd
from os import listdir
from os.path import join
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from sklearn import svm
from sklearn.pipeline import Pipeline
import pylab as pl
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV



img_dir = '../data/uni/'
n_classes = 50
n_files_per_class = 4

def get_img_files(img_dir):
    imgs = filter(lambda x: ".JPG" in x, listdir(img_dir))
    df = pd.DataFrame(index=imgs, columns={'CLASS', 'HOG_DESC', 'TYPE'})
    df["CLASS"] = np.repeat(np.linspace(0, n_classes-1, num=n_classes), n_files_per_class)
    return df


def extract_Pixels(df):
    pix_desc = []
    # Loop over each image
    for img in list(df.index):
        img = Image.open(join(img_dir, img))
        img = np.asarray(list(img.getdata()))
        rows, cols = img.shape
        img_flat = img.reshape(1, rows*cols)
        pix_desc.append(img_flat[0])
    df['PIX_DESC'] = pix_desc
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

    print('Extracting RAW pixel values ...')
    df = extract_Pixels(df)
    # print df


    # KNN
    # Get X, Y
    print('Getting X,Y for training ...')
    df_train = df[df['TYPE']=='TRAIN']

    X_train = list(df_train['PIX_DESC'])
    y_train = list(df_train['CLASS'])


    print 'Extracting Principal Components'
    pca = PCA(n_components=5)
    X_train = pca.fit_transform(X_train)

    print 'Training'
    classifier = KNeighborsClassifier(n_neighbors=3, weights='distance', metric=euclidean)
    # classifier = svm.SVC(kernel='poly', degree=5)
    classifier.fit(X_train, y_train)
    #
    #
    print('Testing ...')
    df_test = df[df['TYPE']=='TEST']

    X_test = list(df_test['PIX_DESC'])
    X_test = pca.transform(X_test)
    y_test = list(df_test['CLASS'])

    print('Accuracy: {}%'.format(classifier.score(X_test, y_test)*100))




