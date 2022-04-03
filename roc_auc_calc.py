# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:00:03 2021

@author: mcalderonloor
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

# Import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# # Binarize the output


y = label_binarize(arr_tuple[:,2], classes=[0, 1, 2, 3])
n_classes = y.shape[1]


x = label_binarize(arr_tuple[:,3], classes=[0, 1, 2, 3])

# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
#                                                     random_state=0)

# # Learn to predict each class against the other
# classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
#                                  random_state=random_state))
# y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# fpra = dict()
# tpra = dict()
# roc_auca = dict()
llsf = []
llst = []
llsr = []

for j in range (0,len(geoid)):
    fpra = dict()
    tpra = dict()
    roc_auca = dict()
    ya = label_binarize(arr_tuple[arr_tuple[:,4]==geoid[j],2], classes=[0, 1, 2, 3])
    n_classesa = ya.shape[1]
    xa = label_binarize(arr_tuple[arr_tuple[:,4]==geoid[j],3], classes=[0, 1, 2, 3])

    for i in range(n_classesa):
        fpra[i], tpra[i], _ = roc_curve(xa[:, i], ya[:, i])
        roc_auca[i] = auc(fpra[i], tpra[i])
    
    llsf.append((fpra))
    llst.append((tpra))
    llsr.append((roc_auca))

lfpr = []
for i in range(0,len(llsf)):
    lfpr.append(llsf[i][1])
lfpr = [item for sublist in lfpr for item in sublist]
lfpr = np.unique(lfpr)
lfpr = lfpr[(lfpr[:] !=0) & (lfpr[:] !=1)] 

# for j in range (geoid):
    
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = roc_curve(x[:, i], y[:, i])
#         roc_auc[i] = auc(fpr[i], tpr[i])


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(x.ravel(), y.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[3], tpr[3], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()