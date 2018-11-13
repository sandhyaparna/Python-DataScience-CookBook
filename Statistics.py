https://docs.scipy.org/doc/scipy/reference/stats.html

### Data Transformation using Box-Cox - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
from scipy import stats
Df['Var_trans'] = stats.boxcox(Df['Var'])[0]

stats.boxcox(Df['Var']) # Produces array of 2 values. 1st value is the transformed data, 2nd value is lmbda found to fit the data to normal
stats.boxcox(Df['Var'],lmbda=value) # Produces array with only 1 value - Transformed data values
stats.boxcox(Df['Var'],alpha=value between 0 & 1) # Produces array with 3 values - Transformed data values, lambda value, (1-alpha)% confidence intervals for lmbda
# After transformation check for Normality using Normality Tests - @ Python-DataScience-CookBook/Hypothesis Tests.py

### Skewness 
from scipy.stats import skew
skew(Df['Var'])

