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

### statsmodels - Without constant term
import statsmodels.api as sm
sm_OLS = sm.OLS(Df_y,Df_X).fit()
Df_y_Pred = sm_OLS.predict(Df_X)
sm_OLS.summary()  #Summary provides coeff, r-sq, adj r-sq etc

### statsmodels - With Constant term (This is similar to what is produced by sklearn)
Df_X_1 = sm.add_constant(Df_X)
sm_OLS_1 = sm.OLS(Df_y,Df_X_1).fit()
Df_y_Pred_1 = sm_OLS_1.predict(Df_X_1)
sm_OLS_1.summary()  #Summary provides coeff, r-sq, adj r-sq etc

### Variables to be created for plots
y_fitted = sm_OLS_1.fittedvalues  # fitted values (need a constant term for intercept)
y_residuals = sm_OLS_1.resid  # model residuals
y_norm_residuals = sm_OLS_1.get_influence().resid_studentized_internal  # normalized residuals
y_norm_residuals_abs_sqrt = np.sqrt(np.abs(y_norm_residuals))   # absolute squared normalized residuals
y_abs_resid = np.abs(y_residuals)   # absolute residuals
y_leverage = sm_OLS_1.get_influence().hat_matrix_diag  # leverage, from statsmodels internals
y_cooks = sm_OLS_1.get_influence().cooks_distance[0]   # cook's distance, from statsmodels internals

import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.rc('figure', titlesize=18)
plt.rc('axes', labelsize=15)
plt.rc('axes', titlesize=18)

# Residual plot
plot_Residual = plt.figure(1)
plot_Residual.set_figheight(8)
plot_Residual.set_figwidth(12)

plot_Residual.axes[0] = sns.residplot(y_fitted, 'Energy', data=Energy, 
                          lowess=True, scatter_kws={'alpha': 0.5},line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
plot_Residual.axes[0].set_title('Residuals vs Fitted')
plot_Residual.axes[0].set_xlabel('Fitted values')
plot_Residual.axes[0].set_ylabel('Residuals')

abs_resid = y_abs_resid.sort_values(ascending=False)
abs_resid_top_3 = abs_resid[:3]

for i in abs_resid_top_3.index:
    plot_Residual.axes[0].annotate(i, xy=(y_fitted[i], y_residuals[i]));
    
## QQ plot
from statsmodels.graphics.gofplots import ProbPlot
QQ = ProbPlot(y_norm_residuals)
plot_QQ = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_QQ.set_figheight(8)
plot_QQ.set_figwidth(12)

plot_QQ.axes[0].set_title('Normal Q-Q')
plot_QQ.axes[0].set_xlabel('Theoretical Quantiles')
plot_QQ.axes[0].set_ylabel('Standardized Residuals');

abs_norm_resid = np.flip(np.argsort(np.abs(y_norm_residuals)), 0)
abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):
    plot_QQ.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], y_norm_residuals[i]));

## Scale-Location Plot
plot_Location = plt.figure(3)
plot_Location.set_figheight(8)
plot_Location.set_figwidth(12)

plt.scatter(y_fitted, y_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(y_fitted, y_norm_residuals_abs_sqrt, 
            scatter=False,  ci=False,  lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_Location.axes[0].set_title('Scale-Location')
plot_Location.axes[0].set_xlabel('Fitted values')
plot_Location.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

abs_sq_norm_resid = np.flip(np.argsort(y_norm_residuals_abs_sqrt), 0)
abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]

for i in abs_norm_resid_top_3:
    plot_Location.axes[0].annotate(i, xy=(y_fitted[i], y_norm_residuals_abs_sqrt[i]));

## Leverage plot
plot_Leverage = plt.figure(4)
plot_Leverage.set_figheight(8)
plot_Leverage.set_figwidth(12)

plt.scatter(y_leverage, y_norm_residuals, alpha=0.5)
sns.regplot(y_leverage, y_norm_residuals, scatter=False, ci=False, lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

plot_Leverage.axes[0].set_xlim(0, 0.02)
plot_Leverage.axes[0].set_ylim(-5, 5)
plot_Leverage.axes[0].set_title('Residuals vs Leverage')
plot_Leverage.axes[0].set_xlabel('Leverage')
plot_Leverage.axes[0].set_ylabel('Standardized Residuals')

leverage_top_3 = np.flip(np.argsort(y_cooks), 0)[:3]

for i in leverage_top_3:
    plot_Leverage.axes[0].annotate(i,xy=(y_leverage[i],y_norm_residuals[i]))
    
def graph(formula, x_range, label=None):
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')

p = len(sm_OLS_1.params) # number of model parameters

graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50), 
      'Cook\'s distance') # 0.5 line
graph(lambda x: np.sqrt((1 * p * (1 - x)) / x), 
      np.linspace(0.001, 0.200, 50)) # 1 line
plt.legend(loc='upper right');



