from __future__ import absolute_import
from __future__ import print_function
from os import listdir
from os.path import join, isfile

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils, generic_utils
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import numpy as np
import cv2


batch_size = 64
data_augmentation = True

nb_classes = 50
nb_epoch = 300
n_files_per_class = 4
img_dir = '../data/uni/'

np.random.seed(1337)  # for reproducibility


def get_img_files(img_dir):
    imgs = filter(lambda x: ".JPG" in x, listdir(img_dir))
    df = pd.DataFrame(index=imgs, columns={'CLASS', 'PIX_DESC', 'TYPE'})
    df["CLASS"] = np.repeat(np.linspace(0, nb_classes - 1, num=nb_classes), n_files_per_class)
    return df

def extract_pixels(df):
    pix_desc = []
    # Loop over each image
    for img in list(df.index):
        img = cv2.imread(join(img_dir, img))
        pix_desc.append(img.astype(np.float32))
    df['PIX_DESC'] = pix_desc
    return df



if __name__ == '__main__':
    print('Reading image files ...')
    df = get_img_files(img_dir)

    print('Separating Training and Test files ...')
    X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(list(df.index), list(df['CLASS']), test_size=0.25, random_state=15)
    df.loc[X_test_file,  'TYPE'] = 'TEST'
    df.loc[X_train_file, 'TYPE'] = 'TRAIN'

    print('Extracting RAW pixel values ...')
    df = extract_pixels(df)

    # Get X, Y
    print('Getting X,Y for training ...')
    df_train = df[df['TYPE']=='TRAIN']

    X_train = np.asarray(list(df_train['PIX_DESC']))
    y_train = np.asarray(list(df_train['CLASS']))

    df_test = df[df['TYPE'] == 'TEST']

    X_test = np.asarray(list((df_test['PIX_DESC'])))
    y_test = np.asarray(list((df_test['CLASS'])))


    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    #
    print(X_train.shape)
    print(X_train.shape[0], 'Xtrain samples')
    print(X_test.shape[0], 'Xtest samples')


    model = Sequential()
    model.add(Dense(960, 505))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(505, 128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, 50))
    model.add(Activation('softmax'))

    opt = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=opt)


    if not data_augmentation:
        print("Not using data augmentation or normalization")
        X_train /= 255
        X_test /= 255
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
        score = model.evaluate(X_test, Y_test, show_accuracy=True)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
    else:
        print("Using real time data augmentation")
        datagen = ImageDataGenerator(
            featurewise_center=True, # set input mean to 0 over the dataset
            samplewise_center=False, # set each sample mean to 0
            featurewise_std_normalization=True, # divide inputs by std of the dataset
            samplewise_std_normalization=False, # divide each input by its std
            zca_whitening=True, # apply ZCA whitening
            rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
            horizontal_flip=True, # randomly flip images
            vertical_flip=True) # randomly flip images

        datagen.fit(X_train)
        print('Preprocessing Over')

        for e in xrange(nb_epoch):
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






