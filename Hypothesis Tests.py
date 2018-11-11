https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/

### Normality Tests
##### Histogram
In repository - Python-DataScience-CookBook/Exploratory Data Analysis.py
import seaborn as sns
sns.distplot(Df.Var.dropna())
##### Q-Q Plot
import numpy as np 
import pylab 
import scipy.stats as stats
stats.probplot(Df.Var, dist="norm", plot=pylab)
pylab.show()
##### Normal Test
k2, p = stats.normaltest(Energy.x)
##### Shapiro-Wilk Test - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
from scipy import stats
w,p = stats.shapiro(Df.x)
##### D’Agostino’s K^2 Test - Kolmogorov-Smirnov test for goodness of fit
stats.kstest(Df.x,'norm')
##### Anderson-Darling Test
stats.anderson((Df.x,'norm')

### Correlation Tests
##### Pearson’s Correlation Coefficient

##### 
##### 
