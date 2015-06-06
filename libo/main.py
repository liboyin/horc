import numpy as np

from classifiers import classify_set, classify_point
from data import C, get_split_df, get_train_x, get_test_x
from scipy.spatial.distance import cdist as euclidean

np.set_printoptions(threshold=np.nan)
num_split = 100
df = get_split_df(num_split)
hist_intersection = lambda x1, x2: np.minimum(x1, x2).sum()
row_mean = lambda x: np.mean(x, axis=0)
tp_counts = np.empty(num_split, dtype=np.uint8)  # number of true predictions in each split
W = 0.35  # point classifier weight
discrimination = list()
for i in range(0, num_split):
    log_prob = classify_set(train_x=get_train_x(df, i, 'sift_kp_descriptors', np.vstack),
                            test_x=get_test_x(df, i, 'sift_kp_descriptors'), cdist=euclidean)
    log_prob += W * classify_point(train_x=get_train_x(df, i, 'hue_histogram', row_mean),
                                   test_x=get_test_x(df, i, 'hue_histogram'), psim=hist_intersection)
    predict_y = np.argmax(log_prob, axis=1)
    tp_flags = np.arange(C) == predict_y
    fp_indices = np.nonzero(np.logical_not(tp_flags))[0]  # fp for false prediction
    print('{} -> {}'.format(fp_indices, predict_y[fp_indices]))  # prints misclassifications
    tp_counts[i] = sum(tp_flags)
    discrimination.append(np.fliplr(np.sort(log_prob, axis=1)).mean(axis=0))
print(tp_counts)
accuracy = tp_counts.astype(np.float32) / C
print('mean_accuracy={}, sigma={}'.format(np.mean(accuracy), np.std(accuracy)))
print(np.array(discrimination).mean(axis=0))
