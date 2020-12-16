# https://docs.scipy.org/doc/scipy/reference/stats.html

### Data Transformation using Box-Cox(Only on Positive data)-Normalization is the process of scaling individual samples to have unit norm
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
from scipy import stats
Df['Var_trans'] = stats.boxcox(Df['Var'])[0]
stats.boxcox(Df['Var']) # Produces array of 2 values. 1st value is the transformed data, 2nd value is lmbda found to fit the data to normal
stats.boxcox(Df['Var'],lmbda=value) # Produces array with only 1 value - Transformed data values
stats.boxcox(Df['Var'],alpha=value between 0 & 1) # Produces array with 3 values - Transformed data values, lambda value, (1-alpha)% confidence intervals for lmbda
# After transformation check for Normality using Normality Tests - @ Python-DataScience-CookBook/Hypothesis Tests.py

### Scaling - Standardization (Std Normal distributed data - mean=0 & variance=1)
from sklearn import preprocessing
Df['Var_scaled'] = preprocessing.scale(Df['Var'])

# Reapply the same transformatioin applied on Train data to Test data
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train) #Scaling is learnt on the train data
scaler.transform(X_train) #Transformation is applied on Train data based on the above
scaler.transform(X_test) #Same Transformation is applied 

# Transforms features by scaling each feature to a given range (Min,Max) 
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
from sklearn import preprocessing
MinMaxScaler = preprocessing.MinMaxScaler(feature_range=(min, max)) #Used for scaling sparse data
# apply transform similar to what is applied on StandardScaler

### Skewness 
from scipy.stats import skew
skew(Df['Var'])

### Outliers
# Z Score - 3 or more standard deviation away from mean
np.abs(stats.zscore(Df['Var']))

### Measure of location or central tendency 
mean = np.average(data) # You can use Pandas dataframe.mean()
weighted_mean = np.average(data, weights=weights)
truncated_mean = stats.trim_mean(data, proportiontocut=0.1)
median = np.median(data) # You can use Pandas dataframe.median()
weighted_median = robustats.weighted_median(x=data, weights=weights)
mode = stats.mode(data)  # You can also use robustats.mode() on larger datasets

### Measure of Variance
import pandas as pd
import numpy as np
from scipy import stats
variance = np.var(data)
standard_deviation = np.std(data)  # df["Population"].std()
mean_absolute_deviation = df["data"].mad()
trimmed_standard_deviation = stats.tstd(data)
median_absolute_deviation = stats.median_abs_deviation(data, scale="normal")  # stats.median_absolute_deviation() is deprecated
# Percentile
Q1 = np.quantile(data, q=0.25)  # Can also use dataframe.quantile(0.25)
Q3 = np.quantile(data, q=0.75)  # Can also use dataframe.quantile(0.75)
IQR = Q3 - Q1




