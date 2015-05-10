__author__ = 'manabchetia'

import pandas as pd
from os import listdir
from os.path import join
from sklearn.cross_validation import train_test_split
import numpy as np
import cv2
from rootsift import RootSIFT
from scipy.spatial.distance import euclidean as euclid_dist
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from scipy.spatial.distance import cosine as cos_dist

img_dir = '../data/uni/'
n_classes = 50
n_files_per_class = 4

def get_img_files(img_dir):
    imgs = filter(lambda x: ".JPG" in x, listdir(img_dir))
    df = pd.DataFrame(index=imgs, columns={'CLASS', 'SIFT_KP', 'SIFT_KP_DESC', 'TYPE'})
    df["CLASS"] = np.repeat(np.linspace(0, n_classes-1, num=n_classes), n_files_per_class)
    return df

def extract_SIFT(df):
    key_points, kp_descriptors = [], []

    # rs = RootSIFT()
    sift = cv2.SIFT(nfeatures=25)

    # Loop over each image
    for img in list(df.index):
        img = cv2.imread(join(img_dir, img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, descs = sift.detectAndCompute(img, None)
        # detector = cv2.FeatureDetector_create("SIFT")
        # kps = detector.detect(img)
        # (kps, descs) = rs.compute(img, kps)
        key_points.append(kps)
        kp_descriptors.append(descs.astype(np.float32))
    df['SIFT_KP'] = key_points
    df['SIFT_KP_DESC'] = kp_descriptors
    return df


def get_X_Y(df):
    X, y = [], []
    for img in list(df.index):
        for dsc in df.loc[img, 'SIFT_KP_DESC']:
            X.append(dsc)
            y.append(df.loc[img, 'CLASS'])
    return X, y


def get_predictions(df, classifier):
    predictions = []
    for img in list(df.index):
        votes = {i:0 for i in xrange(n_classes)}
        # print(votes)
        for dsc in df.loc[img, 'SIFT_KP_DESC']:
            votes[classifier.predict(dsc)[0]] += 1
            # print(votes)
        predictions.append(max(votes, key=votes.get))
        # print(predictions)
    return predictions


def get_predictions2(df, classifier):
    predictions = []
    for img in list(df.index):
        votes = {i:0 for i in xrange(n_classes)}
        for dsc in df.loc[img, 'SIFT_KP_DESC']:
            _, result, _, _ = classifier.find_nearest(dsc, k=5)
            votes[result] += 1
        predictions.append(max(votes, key=votes.get))
    return predictions


def get_accuracy(predictions, truth):
    correct = sum(1 for p,t in zip(predictions, truth) if p==t)
    return correct*100/len(predictions)



if __name__ == '__main__':
    img_dir = '../data/uni/'

    print('Reading image files ...')
    df = get_img_files(img_dir)

    print('Separating Training and Test files ...')
    X_train_file, X_test_file, y_train_file, y_test_file = train_test_split(list(df.index), list(df['CLASS']), test_size=0.25, random_state=42)
    df.loc[X_test_file,  'TYPE'] = 'TEST'
    df.loc[X_train_file, 'TYPE'] = 'TRAIN'

    print('Extracting SIFT features ...')
    df = extract_SIFT(df)

    # Get X, Y
    print('Getting X,Y for training ...')
    df_train = df[df['TYPE']=='TRAIN']
    X_train_dsc, y_train_dsc = get_X_Y(df[df['TYPE']=='TRAIN'])

    print('Training Classifier ...')
    classifier = KNeighborsClassifier(n_neighbors=4, weights='distance', metric=cos_dist)
    # classifier = linear_model.LogisticRegression()
    # classifier = LDA()
    # classifier = QDA()  # Doesn't work
    # classifier = DecisionTreeClassifier()
    # classifier = RandomForestClassifier()
    # classifier = AdaBoostClassifier()
    # classifier = SVC(kernel='poly')
    # classifier = GaussianNB()
    classifier.fit(X_train_dsc, y_train_dsc)




    print('Testing ...')
    df_test = df[df['TYPE']=='TEST']
    predictions = get_predictions(df_test, classifier)


    print('Accuracy: {} %'.format(get_accuracy(predictions, list(df_test['CLASS']))))