from time import time
import utils as ut
import csv
import re
import numpy as np
import os
import argparse
import sys
import itertools


def diagnosis_cut(datadir, testdir):
    # disease codes
    kyw_name = {'MM': ['C2349260', 'C0153869', 'C2349261'],
                'PD': ['C0030567'],
                'T2D': ['C0375115', 'C0375134', 'C0375130',
                        'C0375113', 'C0375119', 'C0375126',
                        'C0375143', 'C23449362', 'C0375151',
                        'C0375137', 'C0375132', 'C0375149',
                        'C0375117', 'C0375122', 'C0375141',
                        'C0375124', 'C0375145', 'C0375139',
                        'C0375147'],
                'AD': ['C0002395'],
                'BC': ['C0153554', 'C0235653', 'C0153551',
                       'C0153553', 'C0024621', 'C0153555',
                       'C0153550', 'C0153552', 'C0260421',
                       'C0153549'],
                'PC': ['C0154088', 'C0376358', 'C0496923',
                       'C0260429']}
    # load ehr data for train and test with AID
    ehr_tr = {}
    with open(os.path.join(testdir,
                           'encodings/cohort-ehr-subseq32-age_in_day.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            ehr_tr.setdefault(r[0], list()).append([int(r[1]), int(r[2])] + list(map(lambda x: str(x), r[3:])))
    ehr_ts = {}
    with open(os.path.join(testdir,
                           'encodings/cohort_test-ehr-subseq32-age_in_day.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            ehr_ts.setdefault(r[0], list()).append([int(r[1]), int(r[2])] + list(map(lambda x: str(x), r[3:])))
    # load scaled convae data for train and test
    enc_tr = {}
    with open(os.path.join(testdir,
                           'encodings/TRconvae_vect_scaled.csv')) as f:
        rd = csv.reader(f)
        for r in rd:
            enc_tr.setdefault(r[0], list()).append(list(map(lambda x: float(x), r[1:])))
    enc_ts = {}
    with open(os.path.join(testdir,
                           'encodings/convae_vect_scaled.csv')) as f:
        rd = csv.reader(f)
        for r in rd:
            enc_ts.setdefault(r[0], list()).append(list(map(lambda x: float(x), r[1:])))
    # store mrn from each disease and vocab terms
    mrn_dis = {}
    dis_idx = {}
    for dis in ut.val_disease:
        with open(os.path.join(datadir,
                               'cohort-{0}.csv'.format(dis))) as f:
            rd = csv.reader(f)
            mrn_dis[dis] = [r[0] for r in rd]

        with open(os.path.join(testdir,
                               'cohort-new-vocab.csv')) as f:
            rd = csv.reader(f)
            idx_to_mt = {}
            next(rd)
            for r in rd:
                idx_to_mt[r[1]] = r[0]
                if (r[0].split('::')[-1] in kyw_name[dis]) and bool(re.match('icd9',
                                                                   r[0].lower())):
                    dis_idx.setdefault(dis, set()).add(str(r[1]))
    # cut encodings
    for dis in mrn_dis.keys():
        diag_idx = {}
        enc_tr_cut = {}
        ehr_tr_cut = {}
        tmp = set()
        for mrn, seq in ehr_tr.items():
            if mrn in mrn_dis[dis]:
                for n, s in enumerate(seq):
                    if len(dis_idx[dis].intersection(set(s[2:]))) != 0:
                        tmp.update(dis_idx[dis].intersection(set(s[2:])))
                        diag_idx[mrn] = (n, s[0])
                        enc_tr_cut[mrn] = enc_tr[mrn][n:]
                        ehr_tr_cut[mrn] = [v[2:] for v in ehr_tr[mrn][n:]]
                        break
        for t in tmp:
            print(idx_to_mt[t])
        print('training')
        print(dis, len(ehr_tr_cut), len(mrn_dis[dis]))  
        with open(os.path.join(datadir,
                               testdir,
                               'encodings/TRconvae-cut-{0}-avg_scaled.csv'.format(dis)),
                               'w') as f:
            wr = csv.writer(f)
            for mrn, vect in enc_tr_cut.items():
                avg = np.mean(vect, axis=0).tolist()
                wr.writerow([mrn] + avg)
        with open(os.path.join(datadir,
                               testdir,
                               'encodings/{0}-TRehr-cut.csv'.format(dis)),
                               'w') as f:
            wr = csv.writer(f)
            for mrn, vect in ehr_tr_cut.items():
                wr.writerow([mrn] + list(itertools.chain(*vect)))
        diag_idx = {}
        enc_ts_cut = {}
        ehr_ts_cut = {}
        for mrn, seq in ehr_ts.items():
            if mrn in mrn_dis[dis]:
                for n, s in enumerate(seq):
                    if len(dis_idx[dis].intersection(set(s[2:]))) != 0:
                        diag_idx[mrn] = (n, s[0])
                        enc_ts_cut[mrn] = enc_ts[mrn][n:]
                        ehr_ts_cut[mrn] = [v[2:] for v in ehr_ts[mrn][n:]]
                        break
        print('test')
        print(dis, len(ehr_ts_cut), len(mrn_dis[dis]))
        with open(os.path.join(datadir,
                               testdir,
                               'encodings/convae-cut-{0}-avg_scaled.csv'.format(dis)),
                               'w') as f:
            wr = csv.writer(f)
            for mrn, vect in enc_ts_cut.items():
                avg = np.mean(vect, axis=0).tolist()
                wr.writerow([mrn] + avg)
        with open(os.path.join(datadir,
                               testdir,
                               'encodings/{0}-ehr-cut.csv'.format(dis)),
                               'w') as f:
            wr = csv.writer(f)
            for mrn, vect in ehr_ts_cut.items():
                wr.writerow([mrn] + list(itertools.chain(*vect)))

"""
Main function
"""


def _process_args():

    parser = argparse.ArgumentParser(
          description='Cut encodings/ehrs from diagnosis for selected diseases')
    parser.add_argument(dest='datadir',
          help='Directory where disease are stored')
    parser.add_argument(dest='testdir',
          help='Test set directory')

    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    args = _process_args()
    print('')
    start = time()

    diagnosis_cut(args.datadir, args.testdir)

    print('\nProcessing time: %s seconds' % round(time() - start, 2))
    print('Task completed\n')

