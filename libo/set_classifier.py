import multiprocessing
import numpy as np

eps = 0.000001
np.set_printoptions(threshold=np.nan)
pool = None

def parallel_map(function, iterable):  # TODO: initializing pool in top level of module leads to namespace issue
    global pool
    if not pool:
        pool = multiprocessing.Pool()
    return pool.map(function, iterable)


def classify_set_step(test_x):  # N' * D
    train_x, cdist = classify_set_cache  # train_x: C * N * D; cdist: R^(A*D), R^(B*D) -> R^(A*B)
    dists = np.array([cdist(test_x, x).min(axis=1) for x in train_x]).sum(axis=1)  # R ^ C
    # for each class in @train_x, sum of min distance from every vector in @test_x to any vector in this class
    probs = dists.max() - dists
    # unnormalized probability defined as the difference from the max distance. Contains at least one zero
    return np.log(eps + np.true_divide(probs, probs.sum()))  # R ^ C, log probability


classify_set_cache = None  # only to be accessed from within @classify_set and @classify_set_step
def classify_set(train_x, test_x, cdist):
    global classify_set_cache
    classify_set_cache = (train_x, cdist)
    result = np.array(parallel_map(classify_set_step, test_x))  # C * C, on @C processes
    classify_set_cache = None
    return result
