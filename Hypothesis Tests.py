https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/

### Quantitative Data
from scipy import stats
##### One Sample T-Test
stats.ttest_1samp(Df.Var,AssumedMean)
##### T-Test on two independent samples
Group1 = Df[Df.Cat_Var=='Cat1']['Var'] # Quantitative Var
Group2 = Df[Df.Cat_Var=='Cat2']['Var'] # Quantitative Var
stats.ttest_ind(Group1, Group2) 
##### T-test on two paired samples
stats.ttest_rel(Df['Var_Before'], Df['Var_After']) 
stats.ttest_1samp((Df['Var_Before'] - Df['Var_After'], 0) 
##### ANOVA
stats.f_oneway(Group1, Group2, Group3, etc) 
##### 

                  
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
w,p = stats.shapiro(Df.Var)
##### D’Agostino’s K^2 Test - Kolmogorov-Smirnov test for goodness of fit
stats.kstest(Df.Var,'norm')
##### Anderson-Darling Test
stats.anderson((Df.Var,'norm')

### Correlation Tests
# H0: Two samples are independent
# H1: There is a dependency between the samples              
##### Pearson’s Correlation Coefficient
corr, p = pearsonr(Df.Var1, Df.Var2)       
##### Spearman’s Rank Correlation
corr, p = spearmanr(Df.Var1, Df.Var2)                    
##### Kendall’s Rank Correlation
corr, p = kendalltau(Df.Var1, Df.Var2)     
               
               
               
               
               
               
               
