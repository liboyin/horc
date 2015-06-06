"""
This script runs a 3 layer Feed Forward Neural Network on GIST features extracted from Images
"""

from __future__ import absolute_import
from __future__ import print_function
from os import listdir
from os.path import join, isfile

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import leargist as gist
from sklearn.cross_validation import train_test_split
from pandas.io.pickle import read_pickle

# GLOBAL VARIABLES
batch_size = 40             # batch size for neural network
nb_classes = 50             # number of classes
nb_epoch = 200              # number of epochs
n_files_per_class = 240     # number of files per class
img_dir = '../data/final'   # path for images
dim_reduction = False       # Set it True if PCA needs to be performed on GIST descriptor

np.random.seed(1500)


def get_img_files(img_dir):
    """
    This function reads the filenames from a directory, assigns labels to the files and saves them to a dataframe
    :param img_dir: path of images
    :return: dataframe
    """
    imgs = filter(lambda x: ".JPG" in x, listdir(img_dir))
    df = pd.DataFrame(index=imgs, columns={'CLASS', 'GIST_DESC', 'TYPE'})
    df["CLASS"] = np.repeat(np.linspace(0, nb_classes - 1, num=nb_classes), n_files_per_class)
    return df


def extract_GIST(df):
    """
    This function extract GIST features from images and saves them in GIST_DESC column of dataframe
    :param df: dataframe containing filename as indices
    :return: dataframe containing GIST features
    """
    gist_desc = []
    # Loop over each image
    for img in list(df.index):
        img = Image.open(join(img_dir, img))
        desc = gist.color_gist(img)
        gist_desc.append(desc.astype(np.float32))
    df['GIST_DESC'] = gist_desc
    return df


def get_accuracy(predictions, truth):
    """
    This function prints the accuracy
    :param predictions: predicted lables
    :param truth: ground thruth
    :return: accuracy in %
    """
    mask = predictions == truth
    correct = np.count_nonzero(mask)
    return correct * 100 / len(predictions)


def get_df(df_cache):
    """
    This function checks if dataframe contain GIST features exists. If it does not exists, it writes the dataframe to a disk
    :param df_cache: filename of dataframe
    :return: dataframe containing GIST descriptors
    """
    if isfile(df_cache):
        print('DataFrame found. \nLoading DataFrame in memory')
        df = read_pickle(df_cache)
    else:
        print('Reading image files ...')
        df = get_img_files(img_dir)

        print('Separating Training and Test files ...')
        # Version 2
        X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(list(df.index), list(df['CLASS']),
                                                                                test_size=0.10, random_state=15)
        df.loc[X_test_file, 'TYPE'] = 'TEST'
        df.loc[X_train_file, 'TYPE'] = 'TRAIN'

        print('Extracting GIST features ...')
        df = extract_GIST(df)

        print('Writing DataFrame to disk')
        df.to_pickle(df_cache)
    return df


if __name__ == '__main__':
    df_cache = 'df.pickle.big'
    df = get_df(df_cache)

    # Getting Training Vectors
    df_train = df[df['TYPE'] == 'TRAIN']

    X_train = np.asarray(list(df_train['GIST_DESC']))
    y_train = np.asarray(list((df_train['CLASS'])))

    # Get Testing vectors
    df_test = df[df['TYPE'] == 'TEST']

    X_test = np.asarray(list((df_test['GIST_DESC'])))
    y_test = np.asarray(list((df_test['CLASS'])))


    # Convert class vectors to binary class matrices for Neural Network
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('Length:', len(Y_train), len(Y_test))

    # Number of Input Nodes in Initial layer of Neural network
    n_input_nodes = 960

    # if PCA is enables
    if dim_reduction:
        principal_components = 300
        pca = PCA(n_components = principal_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        n_input_nodes = principal_components

    # Number of nodes in hidden layer
    n_hidden_nodes = (n_input_nodes + 50)/2


    # Arhitecture of Neural Network
    model = Sequential()
    model.add(Dense(n_input_nodes, 500))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(n_hidden_nodes, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, 50))
    model.add(Activation('softmax'))

    # Optimiser
    opt = RMSprop()
    # opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    # Training Neural Network
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2,
              validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

    print('Test accuracy:', score[1])
