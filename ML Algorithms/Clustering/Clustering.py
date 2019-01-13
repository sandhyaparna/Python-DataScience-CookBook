import numpy as np

cd "C:\Users\Sandhya\OneDrive\Full Time 2016\Projects\Clustering" 

import pandas as pd
Diabetics=pd.read_csv('Diabetics.csv', sep=',')

# Names of Variables in a Data Frame
Diabetics.columns

# Summary Stats of Numeric variables of a Data frame
Diabetics.describe()

# Frequency Table of a single categorical variable
my_tab = pd.crosstab(index=Diabetics["race"],columns="count")
# Freq table - Another way
Diabetics['race'].value_counts()

# Bar chart - Single Categorical variable
my_tab.plot(kind='bar')

# Freq Cross Tab - 2 way Table (2 Categorical - Freq count)
crosstab = pd.crosstab(Diabetics['race'], Diabetics['readmitted'])
# Stacked plot
crosstab.plot(kind='bar', stacked=True, color=['red','blue','green'], grid=False)
# side by side bar plot
crosstab.plot(kind='bar', color=['red','blue','green'], grid=False)

# Data Types
Diabetics.dtypes

# Correlation Matrix
Diabetics.corr()

# Race - (Missing indicator is "?" - Assign it to Other)
# if-then-else statements
Diabetics['race'] = np.where(Diabetics['race']=="?", "Other", Diabetics['race'])

# Filtering Records- 3 obs have unknow or invalid gender - delete those obs
Diabetics = Diabetics[Diabetics['gender']!="Unknown/Invalid"]

# Drop multiple Columns using Column Names
Diabetics = Diabetics.drop(['weight','encounter_id','admission_type_id'],1)

# Drop single column using column Names
# Diabetics = Diabetics.drop('weight',1)

# Drop columns based on column position
Diabetics = Diabetics.drop(Diabetics.columns[[4,5]], axis=1)

# Box plots
import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np
import matplotlib as plt

# Box Plot
Diabetics.boxplot(column='time_in_hospital')

# Box plot by Groups
Diabetics.boxplot(column='time_in_hospital', by = 'readmitted')

# Histograms
Diabetics['time_in_hospital'].hist() 

# Bar Chart (Categoric and Numeric variable)
# Time in Hospital by Race
var = Diabetics.groupby('race').time_in_hospital.mean()
var.plot(kind='bar')

# Line Plot
var.plot(kind='line')

# stacked Column Chart ( 2 categorical vars - by Numeric Var)
var = Diabetics.groupby(['race','readmitted']).time_in_hospital.sum()
var.unstack().plot(kind='bar',stacked=True,  color=['red','blue','green'], grid=False)

# Scatter Plot (num vs num)
plt.pyplot.scatter(Diabetics['num_lab_procedures'],Diabetics['time_in_hospital'])


# Clustering
# sklearn.cluster module is used for clustering

from pandas import Series, DataFrame
import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

# subset clustering variables (Numeric Vars for K-Means)
NumVars=Diabetics[['time_in_hospital','num_lab_procedures','num_procedures','num_medications',
'number_outpatient','number_emergency','number_inpatient','number_diagnoses']]

NumVars.describe()

# standardize clustering variables to have mean=0 and sd=1
Nums=NumVars.copy()
Nums['time_in_hospital']=preprocessing.scale(Nums['time_in_hospital'].astype('float64'))
Nums['num_lab_procedures']=preprocessing.scale(Nums['num_lab_procedures'].astype('float64'))
Nums['num_procedures']=preprocessing.scale(Nums['num_procedures'].astype('float64'))
Nums['num_medications']=preprocessing.scale(Nums['num_medications'].astype('float64'))
Nums['number_outpatient']=preprocessing.scale(Nums['number_outpatient'].astype('float64'))
Nums['number_emergency']=preprocessing.scale(Nums['number_emergency'].astype('float64'))
Nums['number_inpatient']=preprocessing.scale(Nums['number_inpatient'].astype('float64'))
Nums['number_diagnoses']=preprocessing.scale(Nums['number_diagnoses'].astype('float64'))


# split data into train and test sets (70/30 split of data set)
clus_train, clus_test = train_test_split(Nums, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters  (Elbow Curve based on average distance)                                                         
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""
import matplotlib.pylab as plt
plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)

# Attach predicted cluster variable to Train data set
clus_train['cluster'] = clusassign


# plot clusters
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

# Rand Score


# Reference
# https://www.analyticsvidhya.com/blog/2015/07/11-steps-perform-data-analysis-pandas-python/
# https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-learn-data-science-python-scratch-2/
# http://chrisalbon.com/#Python
# http://web.utk.edu/~wfeng1/html/datamining.pdf
# http://file.allitebooks.com/20150823/Learning%20Data%20Mining%20with%20Python.pdf
# Data Visualization - https://www.analyticsvidhya.com/blog/2015/05/data-visualization-python/
# Clustering - https://www.coursera.org/learn/machine-learning-data-analysis/supplement/0y9ri/python-code-k-means-cluster-analysis
# Clustering - http://blog.yhat.com/posts/customer-segmentation-using-python.html


##################### Coursera Example ######################
"""
Created on Mon Jan 18 19:51:29 2016

@author: jrose01
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

"""
Data Management
"""
data = pd.read_csv("tree_addhealth.csv")

#upper-case all DataFrame column names
data.columns = map(str.upper, data.columns)

# Data Management

data_clean = data.dropna()

# subset clustering variables
cluster=data_clean[['ALCEVR1','MAREVER1','ALCPROBS1','DEVIANT1','VIOL1',
'DEP1','ESTEEM1','SCHCONN1','PARACTV', 'PARPRES','FAMCONCT']]
cluster.describe()

# standardize clustering variables to have mean=0 and sd=1
clustervar=cluster.copy()
clustervar['ALCEVR1']=preprocessing.scale(clustervar['ALCEVR1'].astype('float64'))
clustervar['ALCPROBS1']=preprocessing.scale(clustervar['ALCPROBS1'].astype('float64'))
clustervar['MAREVER1']=preprocessing.scale(clustervar['MAREVER1'].astype('float64'))
clustervar['DEP1']=preprocessing.scale(clustervar['DEP1'].astype('float64'))
clustervar['ESTEEM1']=preprocessing.scale(clustervar['ESTEEM1'].astype('float64'))
clustervar['VIOL1']=preprocessing.scale(clustervar['VIOL1'].astype('float64'))
clustervar['DEVIANT1']=preprocessing.scale(clustervar['DEVIANT1'].astype('float64'))
clustervar['FAMCONCT']=preprocessing.scale(clustervar['FAMCONCT'].astype('float64'))
clustervar['SCHCONN1']=preprocessing.scale(clustervar['SCHCONN1'].astype('float64'))
clustervar['PARACTV']=preprocessing.scale(clustervar['PARACTV'].astype('float64'))
clustervar['PARPRES']=preprocessing.scale(clustervar['PARPRES'].astype('float64'))

# split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')

# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()

"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)


# validate clusters in training data by examining cluster differences in GPA using ANOVA
# first have to merge GPA with clustering variables and cluster assignment data 
gpa_data=data_clean['GPA1']
# split GPA data into train and test sets
gpa_train, gpa_test = train_test_split(gpa_data, test_size=.3, random_state=123)
gpa_train1=pd.DataFrame(gpa_train)
gpa_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(gpa_train1, merged_train, on='index')
sub1 = merged_train_all[['GPA1', 'cluster']].dropna()

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

gpamod = smf.ols(formula='GPA1 ~ C(cluster)', data=sub1).fit()
print (gpamod.summary())

print ('means for GPA by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for GPA by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['GPA1'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())



