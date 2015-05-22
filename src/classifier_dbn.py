__author__ = 'manabchetia'


from os import listdir
from os.path import join, isfile
import pandas as pd
import numpy as np
from PIL import Image
import leargist as gist
from sklearn.cross_validation import train_test_split
from sknn.mlp import Classifier, Layer
from nolearn.dbn import DBN
from pandas.io.pickle import read_pickle

# GLOBAL VARIABLES
img_dir = '../data/uni/'
n_classes = 50
n_files_per_class = 4
# n_files_per_class = 240


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
    mask = predictions == truth
    correct = np.count_nonzero(mask)
    return correct * 100 / len(predictions)


def load_df(df_cache):
    # global df
    if isfile(df_cache):
        print('Loading DataFrame in memory')
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
    # img_dir = '../data/uni/'
    # img_dir = '../data/final'

    df_cache = "df.pickle"
    df = load_df(df_cache)


    ## NoLEARN DBN

    # Get X, Y
    print('Getting X,Y for training ...')
    df_train = df[df['TYPE'] == 'TRAIN']

    features_train = np.asarray(list(df_train['GIST_DESC']))
    labels_train = np.asarray(list(df_train['CLASS']))
    print(features_train.shape)
    print(labels_train.shape)
    nn = DBN([features_train.shape[1], 400, 10], learn_rates=0.3, learn_rate_decays=0.9, epochs=10, verbose=1,)

    # print(features_train.shape, labels_train.)
    nn.fit(features_train, labels_train)
    #
    # # Testing
    df_test = df[df['TYPE'] == 'TEST']
    features_test = np.asarray(list(df_test['GIST_DESC']))
    labels_test = list(df_test['CLASS'])
    # print('Accuracy: {}%'.format(nn.score(features_test, labels_test)*100))