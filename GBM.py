# Parameter Tuning
# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/


pd.options.display.max_rows = 1500
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

gblinear booster treats missing values as zeros???

# Shrinkage parameter - Vary shrinkage parameter to check for Test AUC, choose shrinkage parameter that gives highest Test AUC and then
# vary number of trees to avoid over-fitting. Optimal values usually lie between 0.01-0.2









