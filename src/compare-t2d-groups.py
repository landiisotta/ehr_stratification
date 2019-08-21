import csv
from sklearn.metrics import adjusted_mutual_info_score, fowlkes_mallows_score

with open('../data/ehr-804370-test-1/cohort-T2D-innerval-labels.csv') as f:
    rd = csv.reader(f)
    next(rd)
    convae = {}
    convae_mrn = []
    class_lab = []
    for r in rd:
        convae_mrn.append(r[0])
        class_lab.append(r[1])
        convae.setdefault(r[1], list()).append(r[0])

with open('../data/mrn-t2d-groups.csv') as f:
    rd = csv.reader(f)
    next(rd)
    lili = {}
    gt_dict = {}
    for r in rd:
        gt_dict[r[0]] = r[1]
        lili.setdefault(r[1], list()).append(r[0])

gt_lab = [gt_dict[mrn] for mrn in convae_mrn if mrn in gt_dict.keys()]

print("Numerosity T2D dataset LiLi: {0}\n".format(len(gt_dict)))
print("Numerosity T2D dataset CONVAE model: {0}\n".format(len(convae_mrn)))

for g in sorted(lili.keys()):
    print("Group LiLi {0}: {1}\n".format(g, len(lili[g])))
    for cl in sorted(convae.keys()):
        print("Class convae {0}: {1}".format(cl, len(convae[cl])))
        print("Group LiLi {0} -- Class convae {1}: {2}".format(g, cl, len(set(lili[g]).intersection(set(convae[cl])))))
    print("\n")

#print(adjusted_mutual_info_score(gt_lab, class_lab))

print("Fowlkes - Mallows Score: {0:.3f}".format(fowlkes_mallows_score(gt_lab, 
                                                class_lab)))
