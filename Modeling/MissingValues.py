# Number of missing values in each column
Df.isnull().sum()

# Percentage of missing values in each column
Df.isnull().sum() * 100 / len(Df)

# Missing count, Missing Percentage for each column as a dataset - Sorted by count or %
# index consists of variable names
MissingCount = Df.isnull().sum().sort_values(ascending=False)
MissingPercent = (Df.isnull().sum()/X_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([MissingCount, MissingPercent], axis=1, keys=['MissingCount', 'MissingPercent'])
missing_data.reset_index(inplace=True)

# Plot Percentages of missing values










