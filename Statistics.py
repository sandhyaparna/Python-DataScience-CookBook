https://docs.scipy.org/doc/scipy/reference/stats.html

# Data Transformation using Box-Cox
from scipy import stats
Bank['age_trans'] = stats.boxcox(Bank['age'],alpha=0.05)[0]

