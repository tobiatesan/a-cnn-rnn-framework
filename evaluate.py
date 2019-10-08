import os
import numpy as np
from datetime import datetime
import multiprocessing
from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse

import itertools
import numpy as np
np.warnings.filterwarnings('ignore')

NO_THREADS = 4

def argsort2d(m, kind='heapsort'):
    labels = np.argsort(m, kind=kind) # Heapsort is very unstable
    topn_raveled_indices = (labels.ravel() +
                            np.repeat(np.arange(labels.shape[0])*labels.shape[1],
                                      labels.shape[1])).reshape(labels.shape)
    return topn_raveled_indices

def sorted2d(m, argsort):
    return m.ravel()[argsort].reshape(m.shape)

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


def miap(pred, truth, order):
    mgt = pred.shape[1]
    n = pred.shape[0]
    sorted = sorted2d(truth, order)
    R = np.sum(sorted, axis=1)
    assert(R.shape == (n,))
    ranges = np.array_split(range(mgt), NO_THREADS)
    def f(x):
        (ranges, sorted) = x
        return np.stack(np.sum(sorted[:,0:(j+1)], axis=1)*sorted[:,j]/float(j+1) for j in ranges)
    ins = list(zip(ranges,
                itertools.repeat(sorted, NO_THREADS)))
    chunks = list(map(f, ins))
    res = np.nansum(np.concatenate(chunks, axis=0), axis = 0)/R
    assert(res.shape == (n,))
    return np.mean(res)

def map_(pred, truth, order):
    return miap(pred.transpose(), truth.transpose(), order)

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

def evaluate(full_results):
    pred = full_results[:,0,:]
    scores = full_results[:,1,:]
    truth = full_results[:,2,:]
    ######
    full_results = postprocess(full_results)
    ######

    n = full_results.shape[0]
    vchunks = np.array_split(full_results, NO_THREADS, axis = 0)
    hchunks = np.array_split(full_results, NO_THREADS, axis = 2)
    res1 = map(f, vchunks)
    res2 = map(g, hchunks)
    res3 = map(h, vchunks)
    res4 = argsort2d(-scores)
    res5 = argsort2d(-scores.transpose())

    (row_precision, row_recall) = (0,0)
    i = 0
    for (j, chunk) in enumerate(res1):
        (row_precision, row_recall) = (row_precision + chunk[0], row_recall + chunk[1])
        i = i + 1
    row_precision = float(row_precision) / float(i)
    row_recall = float(row_recall) / float(i)

    (class_precision, class_recall) = (0,0)
    i = 0
    for (j, chunk) in enumerate(res2):
        (class_precision, class_recall) = (class_precision + chunk[0], class_recall + chunk[1])
        i = i + 1
    class_precision = float(class_precision) / float(i)
    class_recall = float(class_recall) / float(i)

    (cum_tn, cum_fn, cum_tp, cum_fp) = (0,0,0,0)
    for (tn, fn, tp, fp) in res3:
        cum_tn = cum_tn + tn
        cum_fn = cum_fn + fn
        cum_tp = cum_tp + tp
        cum_fp = cum_fp + fp
    try:
        cumulative_precision = float(cum_tp) / float(cum_tp + cum_fp)
    except ZeroDivisionError as e:
        cumulative_precision = np.nan
    try:
        cumulative_recall = float(cum_tp) / float(cum_tp + cum_fn)
    except ZeroDivisionError as e:
        cumulative_recall = np.nan
    map_i = miap(scores, truth, res4)
    map_c = map_(scores, truth, res5)
    return (map_i, map_c, class_precision, class_recall,
            row_precision, row_recall)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file', help="File with classified examples", type=str)
  args = parser.parse_args()
  results = np.load(args.input_file)
  (map_i, map_c, pcp, pcr, prp, prr) = evaluate(results)
  
  from beautifultable import BeautifulTable
  table = BeautifulTable()
  table.numeric_precision = 4
  table.column_headers = ["Metric", "Per-class", "Per-image"]
  
  table.append_row(["MAP", map_i, map_c])
  table.append_row(["Precision", prp, pcp])
  table.append_row(["Recall", prr, pcr])
  print(table)
