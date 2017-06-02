
# coding: utf-8

# In[1]:

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
get_ipython().magic(u'matplotlib inline')
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns

from sklearn.metrics import mutual_info_score


# Filter bad electrodes
# Take mean across subjects
# 
# 
# Take mean across subjects and classes and big heatmap
# 
# Take mean across all 16 subjects for each invidiual label 
# [1,2,3],[4,5,6]
# 
# 
# replace all bad electrodes w. Nan's and color by them some anomolous color that will stand out 

# In[10]:

electrode_index = np.array(range(0,128))

dist = scipy.io.loadmat('distances.mat')
dist = np.array(dist['distance'])
dist_sorted = np.array([x for (y,x) in sorted(zip(dist,electrode_index))])

allsubjects = np.zeros([16,128,307,480])

for i in range(0,16):
    allsubjects[i] = np.array([(scipy.io.loadmat('s' + str(i + 1) + 'dat.mat')['dat'])])
    allsubjects[i] = allsubjects[i,[dist_sorted]]


# In[49]:

#split dataset based on class into 6 (16x128x307) matricies

subjects_class1 = np.array(allsubjects[:,:,:,0:80])
subjects_class2 = np.array(allsubjects[:,:,:,80:160])
subjects_class3 = np.array(allsubjects[:,:,:,160:240])
subjects_class4 = np.array(allsubjects[:,:,:,240:320])
subjects_class5 = np.array(allsubjects[:,:,:,320:400])
subjects_class6 = np.array(allsubjects[:,:,:,400:480])


# In[50]:

#avg across trials to turn dataset into 16x128x307

subjects_class1 = subjects_class1.mean(axis = 3)
subjects_class2 = subjects_class2.mean(axis = 3)
subjects_class3 = subjects_class3.mean(axis = 3)
subjects_class4 = subjects_class4.mean(axis = 3)
subjects_class5 = subjects_class5.mean(axis = 3)
subjects_class6 = subjects_class6.mean(axis = 3)


# In[51]:

#place each correlation coefficient corresponding to a subject inside 16x128x128 array

totalsubjects = 16
class1_corrcoef = np.zeros([totalsubjects, 128, 128])
class2_corrcoef = np.zeros([totalsubjects, 128, 128])
class3_corrcoef = np.zeros([totalsubjects, 128, 128])
class4_corrcoef = np.zeros([totalsubjects, 128, 128])
class5_corrcoef = np.zeros([totalsubjects, 128, 128])
class6_corrcoef = np.zeros([totalsubjects, 128, 128])


for each_subject in range(0,totalsubjects):
    class1_corrcoef[each_subject] = np.corrcoef(subjects_class1[each_subject])
    class2_corrcoef[each_subject] = np.corrcoef(subjects_class2[each_subject])
    class3_corrcoef[each_subject] = np.corrcoef(subjects_class3[each_subject])
    class4_corrcoef[each_subject] = np.corrcoef(subjects_class4[each_subject])
    class5_corrcoef[each_subject] = np.corrcoef(subjects_class5[each_subject])
    class6_corrcoef[each_subject] = np.corrcoef(subjects_class6[each_subject])


# In[52]:

#average the coefficients across subjects
class1_corrcoef = class1_corrcoef.mean(axis = 0)
class2_corrcoef = class2_corrcoef.mean(axis = 0)
class3_corrcoef = class3_corrcoef.mean(axis = 0)
class4_corrcoef = class4_corrcoef.mean(axis = 0)
class5_corrcoef = class5_corrcoef.mean(axis = 0)
class6_corrcoef = class6_corrcoef.mean(axis = 0)


# In[104]:

fig, (plt1, plt2) = plt.subplots(ncols=2, sharey=True, figsize = (16,6))
sns.heatmap(class1_corrcoef, vmin=-1, vmax=1, ax = plt1, cmap= sns.diverging_palette(240, 10, center = 'dark',as_cmap=True))
sns.heatmap(class4_corrcoef, vmin=-1, vmax=1, ax = plt2, cmap= sns.diverging_palette(240, 10, center = 'dark',as_cmap=True))


# In[105]:

fig, (plt1, plt2) = plt.subplots(ncols=2, sharey=True, figsize = (16,6))
sns.heatmap(class2_corrcoef, vmin=np.min(class2_corrcoef), vmax=np.max(class2_corrcoef), ax = plt1, cmap= sns.diverging_palette(240, 10, center = 'dark',as_cmap=True))
sns.heatmap(class5_corrcoef, vmin=np.min(class5_corrcoef), vmax=np.max(class5_corrcoef), ax = plt2, cmap= sns.diverging_palette(240, 10, center = 'dark',as_cmap=True))


# In[106]:

fig, (plt1, plt2) = plt.subplots(ncols=2, sharey=True, figsize = (16,6))
sns.heatmap(class3_corrcoef, vmin=np.min(class3_corrcoef), vmax=np.max(class3_corrcoef), ax = plt1, cmap= sns.diverging_palette(240, 10, center = 'dark', as_cmap=True))
sns.heatmap(class6_corrcoef, vmin=np.min(class6_corrcoef), vmax=np.max(class6_corrcoef), ax = plt2, cmap= sns.diverging_palette(240, 10, center = 'dark',as_cmap=True))


# In[77]:

#calculates mutual information 

def calc_MI(X,Y,bins):

   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

#returns a 128x128 matrix with mutual informaiton between electrodes

def getMatMI(subject_class):

    A = np.array(subject_class)
    bins = 20 
    n = 128
    matMI = np.zeros((n, n))

    for ix in np.arange(n):
        for jx in np.arange(ix+1,n):
            matMI[ix,jx] = calc_MI(A[:,ix], A[:,jx], bins)
            matMI[jx,ix] = calc_MI(A[:,ix], A[:,jx], bins)
            
    return matMI
        


# In[ ]:

#calc MI for each subject to return 16x128x128 matrix
MI_class1 = np.zeros([16,128,128])
MI_class2 = np.zeros([16,128,128])
MI_class3 = np.zeros([16,128,128])
MI_class4 = np.zeros([16,128,128])
MI_class5 = np.zeros([16,128,128])
MI_class6 = np.zeros([16,128,128])

for each_subject in range(0,16):
    MI_class1[each_subject] = getMatMI(subjects_class1[each_subject])
    MI_class2[each_subject] = getMatMI(subjects_class2[each_subject])
    MI_class3[each_subject] = getMatMI(subjects_class3[each_subject])
    MI_class4[each_subject] = getMatMI(subjects_class4[each_subject])
    MI_class5[each_subject] = getMatMI(subjects_class5[each_subject])
    MI_class6[each_subject] = getMatMI(subjects_class6[each_subject])
    


# In[79]:

#average MI for a given class

MI_class1 = MI_class1.mean(axis = 0)
MI_class2 = MI_class2.mean(axis = 0)
MI_class3 = MI_class3.mean(axis = 0)
MI_class4 = MI_class4.mean(axis = 0)
MI_class5 = MI_class5.mean(axis = 0)
MI_class6 = MI_class6.mean(axis = 0)


# In[103]:

#plot on a heatmap

fig, (plt1, plt4) = plt.subplots(ncols=2, sharey=True, figsize = (16,6))
sns.heatmap(MI_class1, vmin=np.min(MI_class1), vmax=np.max(MI_class1), ax = plt1, cmap= sns.diverging_palette(240, 10,as_cmap=True))
sns.heatmap(MI_class4, vmin=np.min(MI_class4), vmax=np.max(MI_class4), ax = plt4, cmap= sns.diverging_palette(240, 10,as_cmap=True))


# In[107]:

fig, (plt1, plt4) = plt.subplots(ncols=2, sharey=True, figsize = (16,6))
sns.heatmap(MI_class2, vmin=np.min(MI_class2), vmax=np.max(MI_class2), ax = plt1,cmap= sns.diverging_palette(240, 10,as_cmap=True))
sns.heatmap(MI_class5, vmin=np.min(MI_class5), vmax=np.max(MI_class5), ax = plt4,cmap= sns.diverging_palette(240, 10,as_cmap=True))


# In[108]:

fig, (plt1, plt4) = plt.subplots(ncols=2, sharey=True, figsize = (16,6))
sns.heatmap(MI_class3, vmin=np.min(MI_class3), vmax=np.max(MI_class3), ax = plt1,cmap= sns.diverging_palette(240, 10,as_cmap=True))
sns.heatmap(MI_class6, vmin=np.min(MI_class6), vmax=np.max(MI_class6), ax = plt4,cmap= sns.diverging_palette(240, 10,as_cmap=True))


# In[ ]:



