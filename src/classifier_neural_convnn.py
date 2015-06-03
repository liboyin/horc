from __future__ import absolute_import
from __future__ import print_function
from os import listdir
from os.path import join, isfile


import pandas as pd
import numpy as np
from PIL import Image
import leargist as gist
from sklearn.cross_validation import train_test_split
from pandas.io.pickle import read_pickle


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range

batch_size = 32
data_augmentation = True
nb_classes = 50
nb_epoch = 500
n_files_per_class = 240
img_dir = '../data/final'

batch_size = 32
data_augmentation = True

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
                                                                                test_size=0.25, random_state=15)
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

    model.add(Convolution2D(32, 3, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 32, 3, 3, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64*8*8, 512, init='normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, nb_classes, init='normal'))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    print('Model Completed')

    if not data_augmentation:
        print("Not using data augmentation or normalization")

        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")
        X_train /= 255
        X_test /= 255
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=10)
        score = model.evaluate(X_test, Y_test, batch_size=batch_size)
        print('Test score:', score)

    else:
        print("Using real time data augmentation")

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=True, # set input mean to 0 over the dataset
            samplewise_center=False, # set each sample mean to 0
            featurewise_std_normalization=True, # divide inputs by std of the dataset
            samplewise_std_normalization=False, # divide each input by its std
            zca_whitening=False, # apply ZCA whitening
            rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
            horizontal_flip=True, # randomly flip images
            vertical_flip=False) # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        for e in range(nb_epoch):
            print('-'*40)
            print('Epoch', e)
            print('-'*40)
            print("Training...")
            # batch train with realtime data augmentation
            progbar = generic_utils.Progbar(X_train.shape[0])
            for X_batch, Y_batch in datagen.flow(X_train, Y_train):
                loss = model.train(X_batch, Y_batch)
                progbar.add(X_batch.shape[0], values=[("train loss", loss)])

            print("Testing...")
            # test time!
            progbar = generic_utils.Progbar(X_test.shape[0])
            for X_batch, Y_batch in datagen.flow(X_test, Y_test):
                score = model.test(X_batch, Y_batch)
                progbar.add(X_batch.shape[0], values=[("test loss", score)])