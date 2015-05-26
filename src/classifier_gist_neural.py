__author__ = 'manabchetia'

from pyneural import pyneural
from os import listdir
from os.path import join, isfile
import pandas as pd
import numpy as np
from PIL import Image
import leargist as gist
from sklearn.cross_validation import train_test_split
from sknn.mlp import Classifier, Layer
from pandas.io.pickle import read_pickle
from sklearn.externals import joblib
import cPickle

# https://github.com/fchollet/keras/blob/master/examples/mnist_nn.py

# from nolearn.dbn import DBN


# img_dir = '../data/uni/'
img_dir = '../data/final'
n_classes = 50
# n_files_per_class = 4
n_files_per_class = 240
clf_cache = 'pyneural_model_5000' # 240 images per class
# clf_cache = 'pyneural_model_4' # 4 images per class



def get_img_files(img_dir):
    imgs = filter(lambda x: ".JPG" in x, listdir(img_dir))
    df = pd.DataFrame(index=imgs, columns={'CLASS', 'GIST_DESC', 'TYPE'})
    df["CLASS"] = np.repeat(np.linspace(0, n_classes - 1, num=n_classes), n_files_per_class)
    return df


def extract_GIST(df):
    gist_desc = []
    # Loop over each image
    for img in list(df.index):
        img = Image.open(join(img_dir, img))
        desc = gist.color_gist(img)
        gist_desc.append(desc.astype(np.float32))
    df['GIST_DESC'] = gist_desc
    return df


def get_accuracy(predictions, truth):
    mask = predictions==truth
    correct = np.count_nonzero(mask)
    return correct * 100 / len(predictions)


def get_df(df_cache):
    if isfile(df_cache):
        print('DataFrame found. \nLoading DataFrame in memory')
        df = read_pickle(df_cache)
    else:
        print('Reading image files ...')
        df = get_img_files(img_dir)

        print('Separating Training and Test files ...')
        # Version 2
        X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(list(df.index), list(df['CLASS']),
                                                                                test_size=0.25, random_state=15)
        df.loc[X_test_file, 'TYPE'] = 'TEST'
        df.loc[X_train_file, 'TYPE'] = 'TRAIN'

        print('Extracting GIST features ...')
        df = extract_GIST(df)

        print('Writing DataFrame to disk')
        df.to_pickle(df_cache)
    return df


def get_classifier(clf_cache, df, n_iter):
    # global clf
    if isfile(clf_cache):
        print('Model found. \nLoading Model from disk')
        # with open(clf_cache, 'rb') as fid:
        #     clf = cPickle.load(fid)
    else:
        print('Getting X,Y for training ...')
        df_train = df[df['TYPE'] == 'TRAIN']

        features_train = np.asarray(list(df_train['GIST_DESC']))
        labels_train = np.asarray(list(df_train['CLASS']), dtype=np.int8)

        n_rows, n_features = features_train.shape  # 150, 960
        # n_labels = 50

        labels_expanded = np.zeros((n_rows, n_classes), dtype=np.int8)
        for i in xrange(n_rows):
            labels_expanded[i][labels_train[i]] = 1

        print('Training ...')
        clf = pyneural.NeuralNet([n_features, n_iter, n_classes])
        clf.train(features_train, labels_expanded, 10, 40, 0.005, 0.0,
                  1.0)  # features, labels, iterations, batch size, learning rate, L2 penalty, decay multiplier
        # with open(clf_cache, 'wb') as fid:
        #     cPickle.dump(clf, fid)
    return clf


if __name__ == '__main__':
    df_cache = 'df.pickle.big'
    df = get_df(df_cache)

    # PyNeural
    # Get X, Y

    # if isfile(clf_cache):
    clf = get_classifier(clf_cache, df, n_iter=3000)
    joblib.dump(clf, 'filename.pkl')


    print('Testing ...')
    df_test = df[df['TYPE'] == 'TEST']

    features_test = np.asarray(list(df_test['GIST_DESC']))
    labels_test = np.asarray(list(df_test['CLASS']))

    predictions = np.asarray(clf.predict_label(features_test), dtype=np.int8)
    print(predictions)
    print(" ")
    print(labels_test)
    print('Accuracy: {} %'.format(get_accuracy(predictions, labels_test)))


    ## Scikit Neural Network
    # Get X, Y
    # print('Getting X,Y for training ...')
    # df_train = df[df['TYPE'] == 'TRAIN']
    #
    # features_train = np.asarray(list(df_train['GIST_DESC']))
    # labels_train = np.asarray(list(df_train['CLASS']), dtype=np.int8)
    #
    # # Training
    # print("Training ...")
    # nn = Classifier(layers=[Layer("Sigmoid", units=400), Layer("Softmax")], learning_rate=0.001, n_iter=2000)
    # nn.fit(features_train, labels_train)
    #
    # # Testing
    # df_test = df[df['TYPE'] == 'TEST']
    # features_test = np.asarray(list(df_test['GIST_DESC']))
    # labels_test = np.asarray(list(df_test['CLASS']))
    #
    # print('Accuracy: {}%'.format(nn.score(features_test, labels_test)*100))



    ## NoLEARN DBN

    # # Get X, Y
    # print('Getting X,Y for training ...')
    # df_train = df[df['TYPE'] == 'TRAIN']
    #
    # features_train = np.asarray(list(df_train['GIST_DESC']))
    # labels_train = list(df_train['CLASS'])
    # nn = DBN([features_train.shape[1], 400, 10], learn_rates=0.3, learn_rate_decays=0.9, epochs=10, verbose=1,)
    #
    # # print(features_train.shape, labels_train.)
    # nn.fit(features_train, labels_train)
    # #
    # # # Testing
    # df_test = df[df['TYPE'] == 'TEST']
    # features_test = np.asarray(list(df_test['GIST_DESC']))
    # labels_test = list(df_test['CLASS'])
    # # print('Accuracy: {}%'.format(nn.score(features_test, labels_test)*100))