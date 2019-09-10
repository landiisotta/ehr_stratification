from sklearn import preprocessing
from scipy import stats
from sklearn.decomposition import FastICA, TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix, spmatrix, save_npz
from os import path
import deep_patient as dp
from time import time
from datetime import datetime
import numpy as np
import utils as ut
import sys
import argparse
import random
import csv
import os


def run_dp(indir,
           outdir,
           n_dim=ut.dim_baseline,
           test_set=True):

    # load data
    if test_set:
        print('Loading test dataset')
        mrns_ts, raw_data_ts, ehrs_ts, vocab = _load_test_dataset(indir)
        _save_mrns(outdir, mrns_ts, 'bs-mrn.txt')
        print('Loading training dataset')
        mrns, raw_data, ehrs, _ = _load_ehr_dataset(indir)
        _save_mrns(outdir, mrns, 'TRbs-mrn.txt')
    else:
        print('Loading training dataset')
        mrns, raw_data, ehrs, vocab = _load_ehr_dataset(indir)
        _save_mrns(outdir, mrns, 'bs-mrn.txt')

    # run deep patient
    print('Applying DEEP PATIENT')
    if test_set:
        dp_mtx = _deep_patient(raw_mtx, raw_mtx_ts, n_dim)
        _save_matrices(outdir, 'dp-mtx.npy', dp_mtx)
    else:
        dp_mtx = _deep_patient(raw_mtx, raw_mtx, n_dim)
        _save_matrices(outdir, 'dp-mtx.npy', dp_mtx)


"""
Private Functions
"""


# load data
def _load_test_dataset(indir):

    # read the vocabulary
    with open(path.join(indir, 'cohort-new-vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        vcb = {r[1]: r[0] for r in rd}

    with open(path.join(indir, 'cohort_test-new-ehrseq.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        ehrs = {}
        for r in rd:
            ehrs.setdefault(r[0], list()).extend(list(map(int, r[1::])))
    print('Loaded test dataset with {0} patients'.format(
        len(ehrs)))

    # create raw data (scaled) counts
    mrns = [m for m in ehrs.keys()]
    data = ehrs.values()
    raw_dt = np.zeros((len(data), len(vcb)))
    for idx, token_list in enumerate(data):
        for t in token_list:
            raw_dt[idx, t - 1] += 1

    return (mrns, raw_dt, ehrs, vcb)


def _load_ehr_dataset(indir):
    # read the vocabulary
    with open(path.join(indir, 'cohort-new-vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        vcb = {r[1]: r[0] for r in rd}

    # read raw data
    with open(path.join(indir, 'cohort-new-ehrseq.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        ehrs = {}
        for r in rd:
            ehrs.setdefault(r[0], list()).extend(list(map(int, r[1::])))
    print('Loaded dataset with {0} patients and {1} concepts'.format(
        len(ehrs), len(vcb)))

    # create raw data (scaled) counts
    mrns = [m for m in ehrs.keys()]
    data = ehrs.values()
    raw_dt = np.zeros((len(data), len(vcb)))
    for idx, token_list in enumerate(data):
        for t in token_list:
            raw_dt[idx, t - 1] += 1

    return (mrns, raw_dt, ehrs, vcb)


# run deep patient model

def _deep_patient(data_tr, data_ts, n_dim, nvisible=3000):
    param = {'epochs': 5,
             'batch_size': 16,
             'corrupt_lvl': 0.05}
    sda = dp.SDA(nvisible, nhidden=n_dim, nlayer=3, param=param)
    print('Parameters DEEP PATIENT: {0}'.format(param.items()))
    sda.train(data_tr)
    return sda.apply(data_ts)


# save data


def _save_matrices(datadir, filename, data):
    outfile = path.join(datadir, filename)
    np.save(outfile, data)


def _save_mrns(datadir, data, filename):
    with open(path.join(datadir, filename), 'w') as f:
        f.write('\n'.join(data))


"""
Main Function
"""


def _process_args():
    parser = argparse.ArgumentParser(
        description='Baselines')
    parser.add_argument(dest='indir', help='EHR dataset directory')
    parser.add_argument(dest='outdir', help='output directory')
    parser.add_argument('--test_set', dest='test_set', default=False,
                        help='Add fold')
    return parser.parse_args(sys.argv[1:])


if __name__ == '__main__':
    args = _process_args()
    print ('')

    start = time()
    run_dp(indir=args.indir,
           outdir=args.outdir,
           test_set=args.test_set)

    print ('\nProcessing time: %s seconds\n' % round(time() - start, 2))

    print ('Task completed\n')
