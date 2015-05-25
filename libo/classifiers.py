import numpy as np

eps = 0.000001


def classify_set_step(test_x):  # N' * D
    # print('classify_set_step.classify_cache[1]: {}'.format(classify_cache[1]))
    train_x, cdist = classify_cache  # train_x: C * N * D; cdist: R^(A*D), R^(B*D) -> R^(A*B)
    dists = np.array([cdist(test_x, x).min(axis=1) for x in train_x]).sum(axis=1)  # R ^ C
    # for each class in @train_x, sum of min distance from every vector in @test_x to any vector in this class
    probs = dists.max() - dists
    # unnormalized probability defined as the difference from the max distance. Contains at least one zero
    return np.log(eps + np.true_divide(probs, probs.sum()))  # R ^ C, log probability


def classify_point_step(test_x):  # R ^ D
    train_x, psim = classify_cache  # train_x: C * D, psim: R^D, R^D -> R
    sims = np.array([psim(test_x, x) for x in train_x])  # R ^ C
    probs = sims - sims.min()
    # unnormalized probability defined as the difference to the min distance
    return np.log(eps + np.true_divide(probs, probs.sum()))
    # normalized probability defined as the ratio to summed similarities


classify_cache = None  # only to be accessed from within @classify_stem and @classify_step
def classify_stem(train_x, test_x, metric, classify_step):
    global classify_cache
    classify_cache = (train_x, metric)
    result = np.array(map(classify_step, test_x))  # C * C, on @C processes
    classify_cache = None
    return result


classify_set = lambda train_x, test_x, cdist: classify_stem(train_x, test_x, cdist, classify_set_step)
classify_point = lambda train_x, test_x, psim: classify_stem(train_x, test_x, psim, classify_point_step)
