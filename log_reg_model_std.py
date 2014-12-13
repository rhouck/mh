import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import csv
from sklearn.feature_extraction import FeatureHasher
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, roc_auc_score

# convert targets to numpy array
targets = np.genfromtxt('source/ys_prepped.csv', delimiter=',')

# import xs into list of lists
with open('source/xs_prepped.csv', 'rb') as f:
    reader = csv.reader(f)
    xs = list(reader)

# hash rows for input into model
hasher = FeatureHasher(input_type='string', n_features=(2 ** 20))
hashed_rows = hasher.transform(xs)

# split data into train / test sets
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(targets, n_iter=1)
for train_index, test_index in sss:
    train_index = train_index
    test_index = test_index


# confirm test / train / CV sets are appropriately balanced
t = pd.Series(targets)
print "total: clicks - %s, views: %s" % (t[t > 0].shape, t[t == 0].shape)
print ""
 
t = pd.Series(targets[test_index])
print "total - test set: clicks - %s, views: %s" % (t[t > 0].shape, t[t == 0].shape)
t = pd.Series(targets[train_index])
print "total - train set: clicks - %s, views: %s" % (t[t > 0].shape, t[t == 0].shape)
print ""
 
ind = 1
skf = StratifiedKFold(targets[train_index], n_folds=5,)
for cv_train, cv_test in skf:
    t = pd.Series(targets[cv_test])
    print "cv %s - test: clicks - %s, views: %s" % (ind, t[t > 0].shape, t[t == 0].shape)
    t = pd.Series(targets[cv_train])
    print "cv %s - train: clicks - %s, views: %s" % (ind, t[t > 0].shape, t[t == 0].shape)
    print ""
    ind += 1


from sklearn.linear_model import LogisticRegression
from sklearn import grid_search, cross_validation
 
# split train data set into stratified train / test groups X times
# each train / test group conains the full train data set
# stratification aims to maintain the proportion of target values found in full train set in each CV train / test set
logr = LogisticRegression()
parameters = {'C':[.001, .001, .01, .1, 1]}
skf = cross_validation.StratifiedKFold(targets[train_index], n_folds=5,)
 
# high accuracy is proportion of correctly labeled events
# high precision relates to a low false positive rate
# high recall relates to a low false negative rate
scoring = ('accuracy', 'precision', 'recall')
for s in scoring:
    clf = grid_search.GridSearchCV(logr, parameters, cv=skf, scoring=s)
    clf.fit(hashed_rows[train_index], targets[train_index])
    print "Scoring metric: %s" % (s)
    for g in clf.grid_scores_:
        print g
    print ""

 
# test on unseen data
clf = LogisticRegression(C=0.001)
clf.fit(hashed_rows[train_index], targets[train_index])
pred = clf.predict(hashed_rows[test_index])
pred_prob = clf.predict_proba(hashed_rows[test_index])
 
for i in (accuracy_score, precision_score, recall_score):
    print "%s: %0.10f" % (str(i).split()[1], i(targets[test_index], pred))
print "" 
precision, recall, thresholds = precision_recall_curve(targets[test_index], pred_prob[:,1])
prec_recall_chart = pd.DataFrame({
                                'Precision': precision[:-1],
                                'Recall': recall[:-1],
                                }, 
                                index=thresholds)  
prec_recall_chart.plot()
 
auc = roc_auc_score(targets[test_index], pred_prob[:,1])
print "Area Under the Curve (AUC): %0.5f" % (auc)