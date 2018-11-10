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

##### Shapiro-Wilk Test - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html

##### D’Agostino’s K^2 Test

##### Anderson-Darling Test

### Correlation Tests
##### Pearson’s Correlation Coefficient

##### 
##### 
