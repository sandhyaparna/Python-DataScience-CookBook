### Links
# Check for Stationarity: https://medium.com/analytics-vidhya/a-gentle-introduction-to-handling-a-non-stationary-time-series-in-python-8be1c1d4b402
# Check for Stationarity: https://medium.com/@kangeugine/time-series-check-stationarity-1bee9085da05
# https://www.kdnuggets.com/2018/09/end-to-end-project-time-series-analysis-forecasting-python.html

### Steps 
# Look at min and max dates for the data
# Aggregate data to date or week or month level - Do you want to look at overall sales (sum) per day or average sales (mean) per day




# Packages
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


### Plot of Data on Y-axis and Time variable on X-axis


### Check Stationarity
# Split data into two parts and check if the mean and variance are very different from the first & second half of the data or not
X = series.values
split = len(X) / 2
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))

### Time series decomposition into Trend, Seasonal, Residual
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()



