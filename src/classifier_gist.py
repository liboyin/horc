__author__ = 'manabchetia'

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


if __name__ == '__main__':
    img_dir = '../data/uni/'

    print('Reading image files ...')
    df = get_img_files(img_dir)

    print('Separating Training and Test files ...')
    X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(list(df.index), list(df['CLASS']),
                                                                            test_size=0.25, random_state=15)
    df.loc[X_test_file, 'TYPE'] = 'TEST'
    df.loc[X_train_file, 'TYPE'] = 'TRAIN'

    print('Extracting GIST features ...')
    df = extract_GIST(df)

    # Get X, Y
    print('Getting X,Y for training ...')
    df_train = df[df['TYPE'] == 'TRAIN']

    X_train = list(df_train['GIST_DESC'])
    y_train = list(df_train['CLASS'])

    classifier = KNeighborsClassifier(n_neighbors=2, weights='distance', metric=euclidean)
    classifier.fit(X_train, y_train)


    print('Testing ...')
    df_test = df[df['TYPE'] == 'TEST']

    X_test = list(df_test['GIST_DESC'])
    y_test = list(df_test['CLASS'])

    print('Accuracy: {}%'.format(classifier.score(X_test, y_test) * 100))

    predicted_probas = classifier.predict_proba(X_test)
    pp.pprint(predicted_probas)

    # http://scikit-learn.org/stable/modules/pipeline.html
    # http://scikit-learn.org/stable/modules/ensemble.html
    # http://w3facility.org/question/classifiying-a-set-of-images-into-classes/