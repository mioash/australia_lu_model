# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 09:08:07 2021

@author: mcalderonloor
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 14:56:25 2021

@author: mcalderonloor
"""


import gc
from allocate_tiff import allocate
import glob 
from osgeo import gdal
import subprocess
import numpy as np
import pandas as pd


def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows

def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 


def read_all_files(stt,yr1):
   file_names = glob.glob('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/'+str(yr1)+'/'+str(yr1)+'_cf_*')
   #file_names = glob.glob('E:/Marco/australia_lu_model/data/allocation/'+str(stt)+'/'+str(yr1)+'a/'+str(yr1)+'_cf*')
   li = []
   for filename in file_names:
    df = pd.read_csv(filename, index_col=0, header=0)
    li.append(df)
   return li
stt = 'qld'
year = 2015

    
cfms = read_all_files(stt,year)

#cm_all = np.array(cfms[0])
cm_all = pd.DataFrame(cfms[0])

for i in range(1,len(cfms)):
    
    nzero = pd.DataFrame(np.zeros([4,4]))#.astype('i4')
    nzero.columns=['0.0','1.0','2.0','3.0']
    tem1 = pd.DataFrame(cfms[i])
    if '0.0' in tem1.columns:
        nzero['0.0'] = tem1['0.0']
    else:
        nzero['0.0'] = pd.DataFrame(np.zeros([4,1]))
        #nzero[0,:] = pd.DataFrame(np.zeros([1,4]))
        #line = pd.DataFrame({"0.0": 0, "1.0": 0, "2.0" : 0,"3.0": 0}, index=['0'])
        #nzero = pd.concat([nzero.iloc[:0], line, nzero.iloc[0:]]).reset_index(drop=True)
        nzero.iloc[0,:] = 0
    if '1.0' in tem1.columns:
        nzero['1.0'] = tem1['1.0']
    else:
        nzero['1.0'] = pd.DataFrame(np.zeros([4,1]))
        #line = pd.DataFrame({"0.0": 0, "1.0": 0, "2.0" : 0,"3.0": 0}, index=['1'])
        #nzero = pd.concat([nzero.iloc[:1], line, nzero.iloc[1:]]).reset_index(drop=True)
        nzero.iloc[1,:] = 0
    if '2.0' in tem1.columns:
        nzero['2.0'] = tem1['2.0']
    else:
        nzero['2.0'] = pd.DataFrame(np.zeros([4,1]))
        #line = pd.DataFrame({"0.0": 0, "1.0": 0, "2.0" : 0,"3.0": 0}, index=['2'])
        #nzero = pd.concat([nzero.iloc[:2], line, nzero.iloc[2:]]).reset_index(drop=True)
        nzero.iloc[2,:] = 0
    if '3.0' in tem1.columns:
        nzero['3.0'] = tem1['3.0']
    else:
        nzero['3.0'] = pd.DataFrame(np.zeros([4,1]))
        #line = pd.DataFrame({"0.0": 0, "1.0": 0, "2.0" : 0,"3.0": 0}, index=['3.0'])
        #nzero.append(line)
        nzero.loc[3,:] = 0
    nzero = nzero.fillna(0) #[nzero[:,:] ==]
    # nzero['1.0'] = tem1['1.0']
    # nzero['2.0'] = tem1['2.0']
    # nzero['3.0'] = tem1['3.0']
    # #df = pd.DataFrame(np.zeros((2, 3)))
    cm_all = cm_all + nzero #np.array(cfms[int(i)])

cm = cm_all#np.array(cfms[0])
print(accuracy(np.array(cm)))

# for i in range(1,len(cfms)):
    
#     nzero = pd.DataFrame(np.zeros([4,4]))#.astype('i4')
#     nzero.columns=['0','1','2','3']
#     tem1 = pd.DataFrame(cfms[i])
#     if '0' in tem1.columns:
#         nzero['0'] = tem1['0']
#     else:
#         nzero['0'] = pd.DataFrame(np.zeros([4,1]))
#         #nzero[0,:] = pd.DataFrame(np.zeros([1,4]))
#         line = pd.DataFrame({"0": 0, "1": 0, "2" : 0,"3": 0}, index=['0'])
#         nzero = pd.concat([nzero.iloc[:0], line, nzero.iloc[0:]]).reset_index(drop=True)
#     if '1' in tem1.columns:
#         nzero['1'] = tem1['1']
#     else:
#         nzero['1'] = pd.DataFrame(np.zeros([4,1]))
#         line = pd.DataFrame({"0": 0, "1": 0, "2" : 0,"3": 0}, index=['1'])
#         nzero = pd.concat([nzero.iloc[:1], line, nzero.iloc[1:]]).reset_index(drop=True)
#     if '2' in tem1.columns:
#         nzero['2'] = tem1['2']
#     else:
#         nzero['2'] = pd.DataFrame(np.zeros([4,1]))
#         line = pd.DataFrame({"0": 0, "1": 0, "2" : 0,"3": 0}, index=['2'])
#         nzero = pd.concat([nzero.iloc[:2], line, nzero.iloc[2:]]).reset_index(drop=True)
#     if '3' in tem1.columns:
#         nzero['3'] = tem1['3']
#     else:
#         nzero['3'] = pd.DataFrame(np.zeros([4,1]))
#         #line = pd.DataFrame({"0.0": 0, "1.0": 0, "2.0" : 0,"3.0": 0}, index=['3.0'])
#         #nzero.append(line)
#         nzero.loc[3,:] = 0
#     # nzero['1.0'] = tem1['1.0']
#     # nzero['2.0'] = tem1['2.0']
#     # nzero['3.0'] = tem1['3.0']
#     # #df = pd.DataFrame(np.zeros((2, 3)))
#     cm_all = cm_all + nzero 
# # label = [0,1,2,3]

# # for label in range(4):
#     print(f"{label:5d} {recall(label, cm):6.3f}")
    

# print("precision total:", precision_macro_average(cm))




# import sklearn.metrics as metrics
# # calculate the fpr and tpr for all thresholds of the classification
# fpr, tpr, threshold = metrics.roc_curve(arr_tuple[:,3], arr_tuple[:,2])
# roc_auc = metrics.auc(fpr, tpr)

# # method I: plt
# import matplotlib.pyplot as plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()


# def create_arrays(df):
#     # Unstack to make tuples of actual,pred,count
#     df = df.unstack().reset_index()

#     # Pull the value labels and counts
#     actual = df['Actual'].values
#     predicted = df['Predicted'].values
#     totals = df.iloc[:,2].values

#     # Use list comprehension to create original arrays
#     y_true = [[curr_val]*n for (curr_val, n) in zip(actual, totals)]
#     y_predicted = [[curr_val]*n for (curr_val, n) in zip(predicted, totals)]

#     # They come nested so flatten them
#     y_true = [item for sublist in y_true for item in sublist]
#     y_predicted = [item for sublist in y_predicted for item in sublist]

#     return y_true, y_predicted


# import pandas as pd
# import numpy as np
# from sklearn.metrics import confusion_matrix

# labels = ['C', 'F', 'G', 'U']
# df = pd.DataFrame(cm, columns=labels, index=labels)
# df.index.name = 'Actual'
# df.columns.name = 'Predicted'

# # Recreate the original confusion matrix and check for equality
# y_t, y_p = create_arrays(df)
# conf_mat = confusion_matrix(y_t,y_p)
# check_labels = np.unique(y_t)
