from __future__ import absolute_import
from __future__ import print_function
from os import listdir
from os.path import join, isfile

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import pandas as pd
import numpy as np
from PIL import Image
import leargist as gist
from sklearn.cross_validation import train_test_split
from pandas.io.pickle import read_pickle

batch_size = 64
nb_classes = 50
nb_epoch = 300
n_files_per_class = 240
img_dir = '../data/final'

np.random.seed(1337)  # for reproducibility


def get_img_files(img_dir):
    imgs = filter(lambda x: ".JPG" in x, listdir(img_dir))
    df = pd.DataFrame(index=imgs, columns={'CLASS', 'GIST_DESC', 'TYPE'})
    df["CLASS"] = np.repeat(np.linspace(0, nb_classes - 1, num=nb_classes), n_files_per_class)
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
    mask = predictions == truth
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

    df_train = df[df['TYPE'] == 'TRAIN']

    X_train = np.asarray(list(df_train['GIST_DESC']))
    y_train = np.asarray(list((df_train['CLASS'])))

    df_test = df[df['TYPE'] == 'TEST']

    X_test = np.asarray(list((df_test['GIST_DESC'])))
    y_test = np.asarray(list((df_test['CLASS'])))

    #
    print(X_train.shape, 'Xtrain samples')
    print(X_test.shape, 'Xtest samples')


    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('Length:', len(Y_train), len(Y_test))


    model = Sequential()
    model.add(Dense(960, 505))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(505, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, 50))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    # model.compile(loss='mean_squared_error', optimizer=rms)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2,
              validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
