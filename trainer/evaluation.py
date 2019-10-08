import multiprocessing
from datetime import datetime
import tensorflow as tf
import numpy as np
######################
# Model init and loss
######################

from trainer.metrics import compute_per_class, compute_per_row, compute_cumulative, miap, map_
#from trainer.sortutils import *

def f(chunk):
    return compute_per_row(chunk[:,0,:], chunk[:,2,:])

def g(chunk):
    return compute_per_class(chunk[:,0,:], chunk[:,2,:])
def h(chunk):
    return compute_cumulative(chunk[:,0,:], chunk[:,2,:])

def postprocess(full_results, n = 3):
    pred = full_results[:,0,:]
    scores = full_results[:,1,:]
    truth = full_results[:,2,:]
    # -scores for inverse ordering
    topn_raveled_indices = argsort2d(-scores)[:,0:n].ravel()
    pred = np.full(scores.shape, 0)
    pred.ravel()[topn_raveled_indices] = 1
    assert(pred.shape == truth.shape)
    assert(np.all(np.sum(pred, axis=1) == np.repeat(3, pred.shape[0])))
    full_results[:,0,:] = pred
    return full_results

def evaluate(model, sess, handle, preprocess = None):
    test_loss = 0
    test_count = 0
    full_results = None
    while True:
        try:
            predictions, scores, ground, loss = model.eval_psgl_hook(sess, handle)
            test_count += 1
            test_loss += loss
            pack = np.stack((predictions, scores, ground), axis = 1)
            if (full_results is None):
                full_results = pack
            else:
                full_results = np.concatenate((full_results,
                                               pack), axis = 0)
        except tf.errors.OutOfRangeError:
            break
    if (preprocess is not None):
        full_results = preprocess(full_results)
    avgloss = float(test_loss) / float(test_count)
    return (avgloss, full_results)
