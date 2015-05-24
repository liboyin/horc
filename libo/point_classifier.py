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


def classify_point_step(test_x):  # R ^ D
    train_x, psim = classify_point_cache  # train_x: C * D, psim: R^D, R^D -> R
    sims = np.array([psim(test_x, x) for x in train_x])  # R ^ C
    probs = sims - sims.min()
    # unnormalized probability defined as the difference to the min distance
    return np.log(eps + np.true_divide(probs, probs.sum()))
    # normalized probability defined as the ratio to summed similarities


classify_point_cache = None  # only to be accessed from within @classify_set and @classify_set_step
def classify_point(train_x, test_x, psim):
    global classify_point_cache
    classify_point_cache = (train_x, psim)
    result = np.array(parallel_map(classify_point_step, test_x))  # C * C, on @C processes
    classify_point_cache = None
    return result
