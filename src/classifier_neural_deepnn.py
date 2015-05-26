from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
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
import theano



batch_size = 64
nb_classes = 50
nb_epoch = 500

np.random.seed(1337) # for reproducibility



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

df_cache = 'df.pickle.big'
df = get_df(df_cache)

# the data, shuffled and split between tran and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train=X_train.reshape(60000,784)
# X_test=X_test.reshape(10000,784)
# X_train = X_train.astype("float32")
# X_test = X_test.astype("float32")
# X_train /= 255
# X_test /= 255
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

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


print(Y_train.shape, 'Ytrain samples')
print(Y_test.shape, 'Ytest samples')
# print(Y_test)


model = Sequential()
model.add(Dense(960, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 50))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])