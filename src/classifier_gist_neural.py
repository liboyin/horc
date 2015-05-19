__author__ = 'manabchetia'

from pyneural import pyneural
from os import listdir
from os.path import join
import pandas as pd
import numpy as np
from PIL import Image
import leargist as gist
from sklearn.cross_validation import train_test_split
from sknn.mlp import Classifier, Layer


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
    mask = predictions==truth
    correct = np.count_nonzero(mask)
    return correct * 100 / len(predictions)


if __name__ == '__main__':
    img_dir = '../data/uni/'

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

    # # PyNeural
    # Get X, Y
    print('Getting X,Y for training ...')
    df_train = df[df['TYPE'] == 'TRAIN']

    features_train = np.asarray(list(df_train['GIST_DESC']))
    labels_train = np.asarray(list(df_train['CLASS']), dtype=np.int8)

    n_rows, n_features = features_train.shape  # 150, 960
    n_labels = 50

    labels_expanded = np.zeros((n_rows, n_labels), dtype=np.int8)
    for i in xrange(n_rows):
        labels_expanded[i][labels_train[i]] = 1

    print('Training ...')
    nn = pyneural.NeuralNet([n_features, 400, n_labels])
    nn.train(features_train, labels_expanded, 5000, 40, 0.005, 0.0,
             1.0)  # features, labels, iterations, batch size, learning rate, L2 penalty, decay multiplier

    print('Testing ...')
    df_test = df[df['TYPE'] == 'TEST']

    features_test = np.asarray(list(df_test['GIST_DESC']))
    labels_test = np.asarray(list(df_test['CLASS']))

    predictions = np.asarray(nn.predict_label(features_test), dtype=np.int8)
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
