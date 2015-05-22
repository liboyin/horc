__author__ = 'manabchetia'

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
X_train, X_test, y_train, y_test = train_test_split(
    mnist.data / 255.0, mnist.target)

from nolearn.dbn import DBN

clf = DBN(
    [X_train.shape[1], 300, 10],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=1,
    )

clf.fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_score

y_pred = clf.predict(X_test)
print "Accuracy:", zero_one_score(y_test, y_pred)
print "Classification report:"
print classification_report(y_test, y_pred)