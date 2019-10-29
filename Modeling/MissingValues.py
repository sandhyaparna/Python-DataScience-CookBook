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
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data['index'], y=missing_data['MissingPercent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

# Calculate skewness
Df['Var'].skew()

# Median Imputaion
#For Median
meadian_value=train['Age'].median()
train['Age']=train['Age'].fillna(median_value)
# For Mean
mean_value=train['Age'].mean()
train['Age']=train['Age'].fillna(mean_value)
# Mode
data['Gender'].fillna(mode(data['Gender']).mode[0], inplace=True)

# Forward or Backward fill
#for back fill 
train.fillna(method='bfill')
#for forward-fill
train.fillna(method=''ffill)
#one can also specify an axis to propagate (1 is for rows and 0 is for columns)
train.fillna(method='bfill', axis=1)

# MICE - Works only on Numerical Vars
from impyute.imputation.cs import mice
# start the MICE training (Can be applied to all numerical Vars that have missing info in datasets)
Df_NumericalVars = Df.select_dtypes(include = np.number)
Df_Imputed_MICE = pd.DataFrame(data=mice(Df_NumericalVars.values), columns=Df_NumericalVars.columns, index=Df_NumericalVars.index)

# DataWig Imputation - https://github.com/awslabs/datawig - takes a lot of time
import datawig
# Var1 needs to be imputed
# Split data into obs with Var1 not missing and Var1 missing
X_train = X[pd.notnull(X.var1)] #Var1 not missing is used to in training
X_test = X[pd.isnull(X.Var1)] #Var1 missing
# Parameters
imputer = datawig.SimpleImputer(
    input_columns=['Var2','Var3','Var4','Var5','Var6', 'Var7'], # column(s), Categorical & Numerical, these vars themselves can have missing data
    output_column='revol_util', # the column we'd like to impute values for. Can take only 1 column at a time
    output_path = 'imputer_model') # stores model data and metrics
#Fit an imputer model on the train data
imputer.fit(train_df=X_train, num_epochs=50)    #num_epochs is not needed while imputing for Categorical Var (i.e misisng in Cat var)
#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(X_test)

# Soft Probability Imputation
valueCounts = {}
def CountAll():
    global all_columns, nanCounts, valueCounts
    all_columns = list(df)
    nanCounts = df.isnull().sum()
    for x in all_columns:
        valueCounts[x] = df[x].value_counts()

"""Random but proportional replacement(RBPR) of numeric"""
def Fill_NaNs_Numeric(col):

    mini = df[col].min()
    maxi = df[col].max()
    """Selecting ONLY non-NaNs."""
    temp = df[df[col].notnull()][col] # type --> pd.Series

    """Any continuous data is 'always' divided into 45 bins (Hard-Coded)."""
    bin_size = 45
    bins = np.linspace(mini, maxi, bin_size)

    """Filling the bins (with non-NaNs) and calculating mean of each bin."""
    non_NaNs_per_bin = []
    mean_of_bins = []

    non_NaNs_per_bin.append(len(temp[(temp <= bins[0])]))
    mean_of_bins.append(temp[(temp <= bins[0])].mean())
    for x in range(1, bin_size):
        non_NaNs_per_bin.append(len(temp[(temp <= bins[x]) & (temp > bins[x-1])]))
        mean_of_bins.append(temp[(temp <= bins[x]) & (temp > bins[x-1])].mean())

    mean_of_bins = pd.Series(mean_of_bins)
    # np.around() on  list 'proportion' may create trouble and we may get a zero-value imputed, hence,
    mean_of_bins.fillna(temp.mean(), inplace= True)
    non_NaNs_per_bin = np.array(non_NaNs_per_bin)

    """Followoing part is SAME as Fill_NaNs_Catigorical()"""

    """Calculating probability and expected value."""
    proportion = np.array(non_NaNs_per_bin) / valueCounts[col].sum() * nanCounts[col]
    proportion = np.around(proportion).astype('int')

    """Adjusting proportion."""
    diff = int(nanCounts[col] - np.sum(proportion))
    if diff > 0:
        for x in range(diff):
            idx = random.randint(0, len(proportion) - 1)
            proportion[idx] =  proportion[idx] + 1
    else:
        diff = -diff
        while(diff != 0):
            idx = random.randint(0, len(proportion) - 1)
            if proportion[idx] > 0:
                proportion[idx] =  proportion[idx] - 1
                diff = diff - 1

    """Filling NaNs."""
    nan_indexes = df[df[col].isnull()].index.tolist()
    for x in range(len(proportion)):
            if proportion[x] > 0:
                random_subset = random.sample(population= nan_indexes, k= proportion[x])
                df.loc[random_subset, col] = mean_of_bins[x] # <--- Replacing with bin mean
                nan_indexes = list(set(nan_indexes) - set(random_subset))

"""-------------------------------------------------------------------------"""

def Fill_NaNs_Catigorical(col): 
    """Calculating probability and expected value."""
    proportion = np.array(valueCounts[col].values) / valueCounts[col].sum() * nanCounts[col]
    proportion = np.around(proportion).astype('int')
    
    """Adjusting proportion."""
    diff = int(nanCounts[col] - np.sum(proportion))
    if diff > 0:
        for x in range(diff):
            idx = random.randint(0, len(proportion) - 1)
            proportion[idx] =  proportion[idx] + 1
    else:
        diff = -diff
        while(diff != 0):
            idx = random.randint(0, len(proportion) - 1)
            if proportion[idx] > 0:
                proportion[idx] =  proportion[idx] - 1
                diff = diff - 1
        
    """Filling NaNs."""
    nan_indexes = df[df[col].isnull()].index.tolist() 
    for x in range(len(proportion)):
        if proportion[x] > 0:
            random_subset = random.sample(population = nan_indexes, k = proportion[x])
            df.loc[random_subset, col] = valueCounts[col].keys()[x]
            nan_indexes = list(set(nan_indexes) - set(random_subset))



# Using Linear Regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
data = train[['Pclass','SibSp','Parch','Fare','Age']]
#Step-1: Split the dataset that contains the missing values and no missing values are test and train respectively.
x_train = data[data['Age'].notnull()].drop(columns='Age')
y_train = data[data['Age'].notnull()]['Age']
x_test = data[data['Age'].isnull()].drop(columns='Age')
y_test = data[data['Age'].isnull()]['Age']
#Step-2: Train the machine learning algorithm
linreg.fit(x_train, y_train)
#Step-3: Predict the missing values in the attribute of the test data.
predicted = linreg.predict(x_test)
#Step-4: Let’s obtain the complete dataset by combining with the target attribute.
train.Age[train.Age.isnull()] = predicted
train.info()
<class 'pandas.core.frame.DataFrame'>
Int64Index: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            891 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
saleprice      891 non-null int64
dtypes: float64(2), int64(5), object(5)
memory usage: 90.5+ KB
