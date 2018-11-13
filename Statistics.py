https://docs.scipy.org/doc/scipy/reference/stats.html

# Data Transformation using Box-Cox
from scipy import stats
Df['Var_trans'] = stats.boxcox(Df['Var'])[0]

stats.boxcox(Df['Var']) # Produces array of 2 values. 1st value is the transformed data, 2nd value is lmbda found to fit the data to normal
stats.boxcox(Df['Var'],lmbda=value) # Produces array with only 1 value - Transformed data values
stats.boxcox(Df['Var'],alpha=value between 0 & 1) # Produces array with 3 values - Transformed data values, lambda value, (1-alpha)% confidence intervals for lmbda




