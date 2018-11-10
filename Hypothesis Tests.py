https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/

### Normality Tests
##### Histogram
##### Q-Q Plot
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

##### Shapiro-Wilk Test - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html

##### D’Agostino’s K^2 Test

##### Anderson-Darling Test

### Correlation Tests
##### Pearson’s Correlation Coefficient

##### 
##### 
