import utils as ut
import sys
from time import time
import utils
from sklearn.cluster import AgglomerativeClustering
import os
import csv
import argparse
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.sandbox.stats.multicomp import multipletests
from itertools import combinations
import itertools
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster import hierarchy

#Input: ehr lists corresponding to a cluster 
#Output: dictionary of relative term counts
def FreqDict(data, subc, vocab):
    set_data = list(data.values())
    mrgd_data = list(itertools.chain.from_iterable(set_data))
    subc_vect = data[subc]
    unq_mrgd, cnt_mrgd = np.unique(mrgd_data, 
                                   return_counts=True)
    unq_subc, cnt_subc = np.unique(subc_vect, 
                                   return_counts=True)
    all_freq = dict(zip(unq_mrgd, cnt_mrgd))
    freq_dict = dict(zip(unq_subc, cnt_subc))
    new_dict = {t: (freq_dict[t] / all_freq[t],
                     freq_dict[t], all_freq[t])
                for t in freq_dict 
                if vocab[t].split('::')[0] in ut.select_terms}
    return new_dict


#Input: dictionary cluster:ehrs; list mrns
#Output:
def freq_term(data, pred_class, vocab):
    tmp_data = {}
    set_data = {}
    for mrn, subc in pred_class.items():
        tmp_data.setdefault(str(subc), list()).append(data[mrn])
        set_data.setdefault(str(subc), list()).extend(list(set(data[mrn])))
    for subc in range(len(set(tmp_data.keys()))):
        print("Cluster {0} numerosity: {1}".format(subc, len(tmp_data[str(subc)])))
        # percentage of term in subc wrt disease cohort
        rel_term_count = FreqDict(set_data, str(subc), vocab)
        clust_mostfreq = []
        l = 0
        while l < ut.FRpar['n_terms']:
            try:
                MFMT = max(rel_term_count, 
                           key=(lambda key: (rel_term_count[key][1]/len(tmp_data[str(subc)]), rel_term_count[key][0])))
                subc_termc = rel_term_count[MFMT][1]
                num_MFMT = rel_term_count[MFMT][2]
                try:
                    all_comb, p_vals, corrected_p_vals, reject_list, sum_df  = chi_test(raw_ehr, pred_class, MFMT)
                    ctrl_list = [v for idx, v in enumerate(reject_list) if int(subc) in all_comb[idx]]
                    opp_list = [v for idx, v in enumerate(reject_list) if int(subc) not in all_comb[idx]]
                    ctrl_true = len(list(filter(lambda x: x==True, ctrl_list))) > 0
                    #ctrl_true = len(list(filter(lambda x: x==True, ctrl_list))) == len(ctrl_list)
                    ctrl_false = len(list(filter(lambda x: x==False, opp_list))) == len(opp_list)
                    if ctrl_true and ctrl_false:
                        l += 1
                        print("% term:{0} "
                              "= {1:.2f} ({2} terms in cluster ({3:.2f}), "
                              "out of {4} terms in the disease dataset)\n".format(vocab[str(MFMT)], 
                                                                     rel_term_count[MFMT][0], 
                                                                     subc_termc,
                                                                     subc_termc/len(tmp_data[str(subc)]),
                                                                     num_MFMT))
                        for comb, pv, cpv, r in zip(all_comb, p_vals, corrected_p_vals, reject_list):
                            print("Comparison: {0} -- p={1}, corr_p={2}, rej={3}".format(
                                  comb, pv, cpv, r))
                        print(sum_df)
                        print('\n')
                except ValueError:
                    sum_df  = chi_test(raw_ehr, pred_class, MFMT)
                    l += 1
                    print("% term:{0} "
                          "= {1:.2f} ({2} terms in cluster ({3:.2f}), "
                          "out of {4} terms in the disease dataset)\n".format(vocab[str(MFMT)],
                                                                 rel_term_count[MFMT][0],
                                                                 subc_termc,
                                                                 subc_termc/len(tmp_data[str(subc)]),
                                                                 num_MFMT))
                    print(sum_df)
                    print("\n")
                rel_term_count.pop(MFMT)
                clust_mostfreq.append(MFMT)
            except ValueError:
                l += 1
                pass
        print("\n")


##Hierarchical clustering function for outer validation. Max silhouette.
def hclust_ehr(data, mrns, min_cl, max_cl, linkage, affinity):
    list_silh = []
    list_db = [] # Davies-Bouldin index list
    list_ch = [] # Calinski-Harabasz index list
    list_label = []

    # elbow method
    Z = hierarchy.linkage(np.array(data), 'ward')
    last = Z[-max_cl:, 2]
    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters 
    
    min_cl = k
    max_cl = k + 1

    for nc in range(min_cl,max_cl,1):
        hclust = AgglomerativeClustering(n_clusters=nc, 
                                         linkage=linkage, 
                                         affinity=affinity)
        tmp_label = hclust.fit_predict(data).tolist()
        tmp_silh = silhouette_score(data, tmp_label, metric=affinity)
        tmp_db = davies_bouldin_score(data, tmp_label)
        tmp_ch = calinski_harabasz_score(data, tmp_label)
        print("{0} -- Silhouette: {1:.3f}".format(nc, tmp_silh))
        print("DB: {0:.3f}".format(tmp_db))
        print("CH: {0:.3f}".format(tmp_ch))
        list_silh.append(float(tmp_silh))
        list_db.append(float(tmp_db))
        list_ch.append(float(tmp_ch))
        list_label.append(tmp_label)
    best_silh = list_silh.index(max(list_silh))
    best_db = list_db.index(min(list_db))
    best_ch = list_ch.index(max(list_ch))
    print("Number of clusters found:{0}, "
          "Max Silhouette score:{1:.3f}".format(best_silh+min_cl, 
                                    list_silh[best_silh]))
    print("Number of clusters found:{0}, "
          "Min DB score:{1:.3f}".format(best_db+min_cl,
                              list_db[best_db]))
    print("Number of clusters found:{0}, "
          "Max HC score:{1:.3f}\n".format(best_ch+min_cl,
                                list_ch[best_ch]))
    #n_clust = max([best_silh+min_cl, best_db+min_cl, best_ch+min_cl])
    n_clust = best_silh + min_cl
    print("Max number of clusters chosen: {0}".format(n_clust))
    label = list_label[n_clust-min_cl]
    return n_clust, {m: l for m, l in zip(mrns, label)}


# pairwise chi-square test with bonferroni correction
# print only significant comaprisons
def chi_test(data, new_classes, term):
    subc_vect = []
    yes = []
    no = []
    for mrn, subc in new_classes.items():
        subc_vect.append(subc)
        yes.append(list(set(data[mrn])).count(term))
    no = list(map(lambda x: 1-x, yes))
    df = pd.DataFrame()
    df['subcluster'] = subc_vect
    df['0'] = no
    df['1'] = yes
    sum_df = df.groupby(['subcluster']).sum()
    all_comb = list(combinations(sum_df.index, 2))
    p_vals = []
    for comb in all_comb:
        new_df = sum_df[(sum_df.index == comb[0]) | (sum_df.index == comb[1])]
        try:
            chi2, pval, _, _ = chi2_contingency(new_df, correction=True)
        except ValueError:
            return sum_df
        p_vals.append(pval)
    reject_list, corrected_p_vals = multipletests(p_vals, method='bonferroni')[:2]
    return all_comb, p_vals, corrected_p_vals, reject_list, sum_df

# pairwise ttest for confounders age and seq_len
def pairwise_ttest(val_vec, cnf):
    df = pd.DataFrame()
    cluster = []
    score = []
    for subc, dic_conf in val_vec.items():
        cluster += [str(subc) for idx in range(len(dic_conf[cnf]))]
        score.extend(dic_conf[cnf])
    df['subcluster'] = cluster
    df['score'] = score
#    all_comb = list(combinations(df.subcluster, 2))
#    p_vals = []
#    for comb in all_comb:
#        g1 = df[(df.subcluster == comb[0])]['score'] 
#        g2 = df[(df.subcluster == comb[1])]['score']
#        stat, pval = ttest_ind(g1, g2, equal_var=False)
#        p_vals.append(pval)
#    reject_list, corrected_p_vals = multipletests(p_vals, method='bonferroni')[:2]
#    for comb, pv, cpv, r in zip(all_comb, p_vals, corrected_p_vals, reject_list):
#        print("Comparison: {0} -- p={1}, corr_p={2}, rej={3}".format(
#              comb, pv, cpv, r))
    MultiComp = MultiComparison(df['score'],
                                df['subcluster'])
    comp = MultiComp.allpairtest(ttest_ind, 
                                 method='bonf')
    print(comp[0])
    pd.options.display.float_format = '{:.3f}'.format
    print(df.groupby(['subcluster']).describe())


# pairwise chi-sq test for confounder sex
def pairwise_chisq(val_vec, cnf):
    cluster = list(val_vec.keys())
    fem = [val_vec[subc][cnf].count('Female') 
           for subc in val_vec.keys()]
    mal = [val_vec[subc][cnf].count('Male')
           for subc in val_vec.keys()]
    df = pd.DataFrame()
    df['female'] = fem
    df['male'] = mal
    df.index = cluster
    all_comb = list(combinations(df.index, 2))
    p_vals = []
    for comb in all_comb:
        new_df = df[(df.index == comb[0]) | (df.index == comb[1])]
        chi2, pval, _, _ = chi2_contingency(new_df, correction=True)
        p_vals.append(pval)
    reject_list, corrected_p_vals = multipletests(p_vals, method='bonferroni')[:2]
    for comb, pv, cpv, r in zip(all_comb, p_vals, corrected_p_vals, reject_list):
        print("Comparison: {0} -- p={1:.3f}, corr_p={2:.3f}, rej={3}".format(
              comb, pv, cpv, r))
    print(df)


# Check for confounders between subclusters (i.e. SEX, AGE, SEQ_LEN)
def check_confounders(datadir, raw_ehr, label):
    with open(os.path.join(datadir, 'patient-details.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        p_details = {}
        for r in rd:
            if r[0] in list(label.keys()):
                p_details[r[0]] = [float(r[1]), r[2], len(raw_ehr[r[0]])]
    n_cl = len(set(label.values()))
    val_vec = {}
    for mrn, subc in label.items():
        if subc in val_vec:
            val_vec[subc]['age'].append(p_details[mrn][0])
            val_vec[subc]['sex'].append(p_details[mrn][1])
            val_vec[subc]['seq_len'].append(p_details[mrn][2])
        else:
            val_vec[subc] = {'age':[p_details[mrn][0]],
                             'sex': [p_details[mrn][1]],
                             'seq_len': [p_details[mrn][2]]}
    print("Multiple comparison t-test for AGE:")
    pairwise_ttest(val_vec, 'age')
    print("\nMultiple comparison t-test for SEQUENCE LENGTH:")
    pairwise_ttest(val_vec, 'seq_len')
    print("\nMultiple comparison chi-square test for SEX:")
    pairwise_chisq(val_vec, 'sex')


##Internal clustering validation
"""
data: convae output (mrn, avg_vect)
raw_ehr: (mrn, term_seq)
mrn_dis: [m for m in mrn with disease]
"""
def inner_clustering_analysis(disease, data, mrns, raw_ehr,
                              min_cl, max_cl, linkage, affinity, 
                              vocab):
    print("Cohort {0} numerosity: {1}\n".format(disease, len(data)))
    n_clust, mrn_label = hclust_ehr(data, mrns, min_cl, max_cl, linkage, affinity)
    freq_term(raw_ehr, mrn_label, vocab)
    return mrn_label


"""
Main function
"""


def _process_args():

    parser = argparse.ArgumentParser(
          description='Inner clustering encodings evaluation')
    parser.add_argument(dest='datadir',
          help='Directory where disease are stored')
    parser.add_argument(dest='indir',
          help='Directory with complete ehrs')
    parser.add_argument(dest='encdir',
          help='Directory with encodings')
    parser.add_argument(dest='disease',
          help='Specify disease acronym')

    return parser.parse_args(sys.argv[1:])


if __name__ == "__main__":
    args = _process_args()
    print('')
    start = time()

    scaler = MinMaxScaler()

    with open(os.path.join(args.datadir, 
                           'cohort-{0}.csv'.format(args.disease))) as f:
        rd = csv.reader(f)
        mrns_dis = [str(r[0]) for r in rd]
    with open(os.path.join(args.encdir, 
                           'convae-cut-{0}-avg_scaled.csv'.format(args.disease))) as f:
        rd = csv.reader(f)
        data = {}
        for r in rd:
            if r[0] in mrns_dis:
                data[r[0]] = r[1::]
    mrns = list(data.keys())
    data = np.array(list(data.values())).astype(np.float64)
    data = scaler.fit_transform(data)
    with open(os.path.join(args.encdir, 
                           '{0}-ehr-cut.csv'.format(args.disease))) as f:
        rd = csv.reader(f)
        raw_ehr = {}
        for r in rd:
            if r[0] in mrns_dis and r[0] in mrns:
                raw_ehr[str(r[0])] = r[1::]
    with open(os.path.join(args.indir, 
                           'cohort-new-vocab.csv')) as f:
        rd = csv.reader(f)
        next(rd)
        idx_to_mt = {r[1]: r[0] for r in rd}

    label = inner_clustering_analysis(args.disease, data,
                                      mrns, 
                                      raw_ehr, 
                                      ut.HCpar['min_cl'],
                                      ut.HCpar['max_cl'], 
                                      ut.HCpar['linkage_clu'],
                                      ut.HCpar['affinity_clu'],
                                      idx_to_mt)

    with open(os.path.join(args.indir, 
         'cohort-{0}-innerval-labels.csv'.format(args.disease)), 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["MRN", "OUT-CL"])
        for mrn, lab in label.items():
            wr.writerow([mrn, lab])

    check_confounders(args.datadir, raw_ehr, label)

    print('\nProcessing time: %s seconds' % round(time()-start, 2))
    print('Task completed\n')

