__author__ = 'manabchetia'

from pyneural import pyneural
# import neurolab
from os import listdir
from os.path import join
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
from PIL import Image
import leargist as gist
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import euclidean
from sklearn.externals import joblib
import pprint as pp


img_dir = '../data/uni/'
n_classes = 50
n_files_per_class = 4


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
    correct = sum(1 for p, t in zip(predictions, truth) if p == t)
    return correct * 100 / len(predictions)

if __name__ == '__main__':
    img_dir = '../data/uni/'

    print('Reading image files ...')
    df = get_img_files(img_dir)

    print('Separating Training and Test files ...')
    # X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(list(df.index), list(df['CLASS']),
    #                                                                         test_size=0.25, random_state=15)
    # df.loc[X_test_file, 'TYPE'] = 'TEST'
    # df.loc[X_train_file, 'TYPE'] = 'TRAIN'

    print('Extracting GIST features ...')
    df = extract_GIST(df)

    # Get X, Y
    # print('Getting X,Y for training ...')
    # df_train = df[df['TYPE'] == 'TRAIN']

    features = np.asarray(list(df['GIST_DESC']))
    labels = np.asarray(list(df['CLASS']), dtype=np.int8)

    n_rows, n_features = features.shape  # 150, 960
    n_labels = 50

    labels_expanded = np.zeros((n_rows, n_labels), dtype=np.int8)
    for i in xrange(n_rows):
        labels_expanded[i][labels[i]] = 1


    print('Training ...')
    nn = pyneural.NeuralNet([n_features, 400, n_labels])
    nn.train(features, labels_expanded, 1000, 40, 0.01, 0.0, 1.0) # features, labels, iterations, batch size, learning rate, L2 penalty, decay multiplier


    print('Testing ...')


    predictions = nn.predict_label(features)
    print(predictions)
    print(" ")
    print(labels)
    print('Accuracy: {} %'.format(get_accuracy(predictions, list(df['CLASS']))))
