pd.options.display.max_rows = 1500
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

import sklearn as sklearn
from sklearn import *
from sklearn.linear_model import *
from sklearn.model_selection import *

# Extract Numeric independent Vars
Df_X = Df[['Var1','Var2','Var3']]

# Numeric Target Variable
Df_y = Df['Target_Var']

### Sklearn
lm = linear_model.LinearRegression()
lm = lm.fit(Df_X, Df_y)
Df_y_Pred = lm.predict(Df_X)
R_squared = lm.score(Df_X, Df_y)   # R-squared
Adjusted_r_squared = 1 - (1-R_squared)*(len(Df_y)-1)/(len(Df_y)-Df_X.shape[1]-1)  # Adjusted R-squared
lm.coef_
lm.intercept_







