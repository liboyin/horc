Dependencies:
OpenCV (2.4.11)
Python (2.7.5)
Numpy (1.9.2)
Scipy (0.15.1)
Pandas (0.16.1)

Dataset:
The dataset is stored in a folder called "data", located in the same directory as the source code. Images in the folder are named such that the first M images belong to class 0, the second M images belong to class 1, etc. There are C classes in total. The value of M and C can be changed in data.py.

main.py
Main.py tests the classifier on 100 random splits. The code outputs misclassifications in each step, as well as the accuracy and standard deviation of classification.

test.py:
Test.py classifies a new image, using all images in the given dataset as training data. The path to the new image is in line 16. The code outputs the C-dimensional unnormalized log probability vector, as well as the classification result.