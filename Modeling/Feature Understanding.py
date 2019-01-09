# Delete noisy features - where Trend_Correlation<0.95 and where there are unequal number of trend changes
from featexp import get_trend_stats
stats = get_trend_stats(data=data_train, target_col='target', data_test=data_test)


