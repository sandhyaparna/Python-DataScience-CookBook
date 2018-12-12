# Binning helps reducing the noise or non-linearity. Allows easy identification of outliers, invalid and missing values of numerical variables 
# Do not use target class info during Binning 

# http://www.saedsayad.com/binning.htm

### Equal Width Binning
# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.cut.html
# k is number of bins to be created
# Width of each bin = (Max-Min)/k
Df['Var_Bin'] = pd.cut(Df['Num_Var'],k) 

### Equal Freq binning
# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.qcut.html
# q is number of quantiles or array of quantiles
# q is 10 for deciles, 4 for quantiles etc; array of quantiles:[0, .25, .5, .75, 1.] for quartiles
Df['Var_Bin'] = pd.cut(Df['Num_Var'],q) 

### Entropy Based Binning
# http://www.saedsayad.com/unsupervised_binning.htm
# https://github.com/paulbrodersen/entropy_based_binning






