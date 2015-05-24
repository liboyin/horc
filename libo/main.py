import numpy as np

# from classifiers import classify_set, classify_point
from set_classifier import classify_set
from point_classifier import classify_point

from data import C, get_split_df, get_train_x, get_test_x
from scipy.spatial.distance import cdist as euclidean

num_split = 10
df = get_split_df(num_split)
hist_intersection = lambda x1, x2: np.minimum(x1, x2).sum()
row_mean = lambda x: np.mean(x, axis=0)
tp_counts = np.empty(num_split, dtype=np.uint8)  # number of true predictions in each split
W = 0.5  # point classifier weight
for i in range(0, num_split):
    log_prob = classify_set(train_x=get_train_x(df, i, 'sift_kp_descriptors', np.vstack),
                            test_x=get_test_x(df, i, 'sift_kp_descriptors'), cdist=euclidean)
    log_prob += W * classify_point(train_x=get_train_x(df, i, 'hue_histogram', row_mean),
                                   test_x=get_test_x(df, i, 'hue_histogram'), psim=hist_intersection)
    print(log_prob)
    predict_y = np.argmax(log_prob, axis=0)
    tp_flags = np.arange(C) == predict_y
    fp_indices = np.nonzero(np.logical_not(tp_flags))[0]  # fp for false prediction
    print('{} -> {}'.format(fp_indices, predict_y[fp_indices]))
    tp_counts[i] = sum(tp_flags)
print(tp_counts)
accuracy = tp_counts.astype(np.float32) / C
print('mean_accuracy={}, sigma={}'.format(np.mean(accuracy), np.std(accuracy)))
# TODO: extract sift from the hue channel?
