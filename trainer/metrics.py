from sklearn.metrics import confusion_matrix
import numpy as np

def compute_per_axis(pred, truth, axis):
    pos = np.sum(pred, axis=axis)
    neg = pred.shape[axis] - pos
    tp = np.sum(np.logical_and(pred, truth), axis=axis)
    tn = np.sum(np.logical_not(np.logical_or(pred, truth)), axis=axis)
    fp = pos - tp
    fn = neg - tn
    per_ax_prec = tp / (tp + fp)
    per_ax_rec = tp / (tp + fn)
    ax_precision = np.nanmean(per_ax_prec, axis=0)
    ax_rec = np.nanmean(per_ax_rec, axis=0)
    return ax_precision, ax_rec

def compute_per_class(pred, truth):
    return compute_per_axis(pred, truth, 0)

def compute_per_row(pred, truth):
    return compute_per_axis(pred, truth, 1)

def compute_cumulative(pred, truth):
    pos = np.sum(np.sum(pred, axis=1), axis=0)
    neg = pred.shape[0]*pred.shape[1] - pos
    tp = np.sum(np.sum(np.logical_and(pred, truth), axis=1), axis=0)
    tn = np.sum(np.sum(np.logical_not(np.logical_or(pred, truth)), axis=1), axis=0)
    fp = pos - tp
    fn = neg - tn
    return (tn, fn, tp, fp)
#from trainer.sortutils import *

def miap(pred, truth):
    mgt = pred.shape[1]
    n = pred.shape[0]
    order = argsort2d(-pred)
    sorted = sorted2d(truth, order)
    R = np.sum(sorted, axis=1)
    assert(R.shape == (n,))
    intermediates = np.stack((np.sum(sorted[:,0:(j+1)], axis=1)*sorted[:,j]/float(j+1)) for j in range(mgt))
    res = np.nansum(intermediates, axis = 0)/R
    assert(res.shape == (n,))
    return np.mean(res)

def map_(pred, truth):
    return miap(pred.transpose(), truth.transpose())

def mean_average_precision_at_n(pred, truth, n):
    pass
