### CheatSheets ###
# https://www.analyticsvidhya.com/blog/2015/06/infographic-cheat-sheet-data-exploration-python/
# http://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PythonForDataScience.pdf
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Pandas_Cheat_Sheet_2.pdf

################# Miscellaneous #################
# Display more data in Terminal/Console
pd.options.display.max_rows = 1500


################# DATA LOADING / IMPORTING #################
### Preview of a file-csv/text before importing as a Df
list(open('path/file.csv'))
list(open('path/file.txt'))

### SQL Server - import files/tables directly 
import pip
pip.main(['install','--upgrade','pyodbc']) # old versio pip
import subprocess
subprocess.check_call(["python", '-m', 'pip', 'install', 'pyodbc'])
import pyodbc 
Connection = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=vigi-dw01;" #Change
                      "Database=Datawarehouse;" #Change
                      "Trusted_Connection=yes;")
SQLCommand = ('select * from [DataWarehouse].[dbo].[DimDate]') # SQLQuery
Df = pd.read_sql_query(SQLCommand, Connection)   # Returns data frame   

### CSV files 
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
Df = pd.read_csv("path/file.csv",
                 dtype = {'Var1': str}, # Change data type of a variable
                 parse_dates = ['Var2'], # To import variable in datetime format default format is '%d%b%Y:%H:%M:%S.%f'
                 usecols = ['Var1', # Import specific columns
                            'Var2',
                            'Var5',
                            'Var8']) 

### TEXT files - Tab Delimited is default 
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_table.html
Df = pd.read_table("path/file.txt", 
                    dtype = {'Var1': str}, # Change data type of a variable
                    parse_dates = ['Var2'], # To import variable in datetime format default format is '%d%b%Y:%H:%M:%S.%f'
                    sep='\t') # Tab Delimited 
header=None -  When header is not present and also assigns default column names
names=['Var1', 'Var2', 'Var3'] - When header is not present and u want to assign names
index_col='Var2' - Var2 to be assigned as index column of the dataframe (Var2 will not be part of Variables within the dataframe anymore)
index_col=['key1', 'key2'] - Hierarchical index from multiple columns (Similar to groupby)
sep='\s+' - When there is whitespace or some other pattern to separate fields
skiprows=[0, 2, 3] - skip specified rows

### TEXT files - Pipe Delimited 
Df = pd.read_table("path/file.txt", 
                    dtype = {'Var1': str}, # Change data type of a variable
                    parse_dates = ['Var2'], # To import variable in datetime format default format is '%d%b%Y:%H:%M:%S.%f'
                    sep='|') # Pipe Delimited 

# read_clipboard - Version of read_table that reads data from the clipboard. Useful for converting tables from web pages
# read_fwf - Read data in fixed-width column format (that is, no delimiters)


################# SAVE DATA / EXPORTING #################
### Export dataframes/tables to SQL Servere 
???

### Export CSV files 
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html
Df.to_csv('path/NewFile.csv',index=False) # Index=False for no rownames in the exported file

### Export TEXT files - Tab Delimited 
Df.to_csv('path/NewFile.txt', sep='\t', index=False)

### Export TEXT files - Pipe Delimited
Df.to_csv('path/NewFile.txt', sep='|', index=False)


################# Descriptive Stats #################
# Size of a dataframe
print('This dataset has ' + str(Df.shape[0]) + ' rows, and ' + str(Df.shape[1]) + ' columns')

# Column names
list(Df)

# Data Types
Df.dtypes
# Data type of a single var
Df['Var1'].dtype

# Non-missing values in each column, data type of vars
Df.info()
# If number of columns exceeds default max cols - output is truncated - specify max_cols for each column analysis
Df.info(max_cols=500)

# Summary of only Numeric variables/columns - Number of records in a column, quantiles, mean, std
Df.describe()
# Summary of only Character variables
Df.describe(include=[np.object])
  Unique - Number of uniques values
  top - Highest frequent value within a var - If highest freq is same for 2 or more values, only only value of them is displayed in top
  freq - Highest frequent value's freq
np.bool for boolean vars
# Summary of all variables - object,char,bool,int,float
Df.describe(include = 'all')

# For Specific percentiles
Df.describe(percentiles=[.1, .15, .3, .7])

# Mean of all vars in df
Df.mean(axis=0)

# Mean of all rows in df
Df.mean(axis=1)

# Number of missing values in each column
Df.isnull().sum()
Df['Var1'].isnull().sum()

# First few rows sample - Glimpse of data
Df.head()
Df['Var1'].head()

# Last few rows - Glimpse of data
Df.tail()
Df['Var1'].tail()

# Freq Dist
pd.value_counts(Df['Var1'].values)

# Unique values
pd.unique(Df['Var1'])
df['Var1'].nunique()

# Mean absolute deviation
Df['Var1'].mad()

# Median absolute deviation
from statsmodels.robust.scale import mad
mad(Df['Var1'], c=1)

### Quantiles
# 10 represents 10 quantiles
pd.qcut(df.var,10,labels=False,retbins=True)
# Percentiles or Quantiles
np.percentile(Df.Var, 50)
Df.Var.quantile([0.1,0.2,0.3,0.9])
# If there are missing values in data - but get percentiles
np.nanpercentile(Df.Var, 50)

### Outliers - Detect and remove - http://colingorrie.github.io/outlier-detection.html
# If you have multiple columns in your dataframe and would like to remove all rows that have outliers in at least one column (remove rows with numerical data above 3 std
# keep only the ones that are within +3 to -3 standard deviations
New_Df = Df[(np.abs(stats.zscore(Df[['Num_Var1','Num_Var2','Num_Var3']])) < 3).all(axis=1)]
# Filter based on z-score of a single numeric variable
df[np.abs(df.Num_Var1-df.Num_Var1.mean()) <= (3*df.Num_Var1.std())]
# same as above code
df[np.abs((df.Num_Var1-df.Num_Var1.mean())/(df.Num_Var1.std())) <= (3)]
# Based on quantiles - Identify the 99% quantile of a NumVar & then filter the data based on that value
df[df["col"] < (df["col"].quantile(0.99))]
# Based on modified z-score
from statsmodels.robust.scale import mad
df[np.abs( (0.6745 * (df.Num_Var1-df.Num_Var1.median()))/(mad(df['Num_Var1'], c=1)) ) <= (3)]

# Cross tab
pd.crosstab(Df['Var1'],Df['Var2'])

# Cross tab - 3vars
Var3 unique value counts within group of (Var1,Var2)
pd.crosstab(Df.Var3, [Df.Var1, Df.Var2])

# Within unique values of Var(Cat) - give sum of values in Var1(Numeric)
Df.groupby(['Var']).Var1.sum()
# Plotting of a bar graph of the same
Df.groupby('Var').Var1.sum().plot.bar()

# Summary stats of Var3 as a group by Var1,Var2 - Different from cross tabs - crosstabs gives counts by each unique value with Var3
# where as this gives summary stats of the entire Var3 within that grouping
Df.groupby(['Var1', 'Var2']).Var3.describe().unstack()

################# Visualization #################
### https://python-graph-gallery.com/

### Univariate Analysis of Numeric variables
# seaborn - https://seaborn.pydata.org/examples/index.html
# Kde - https://seaborn.pydata.org/generated/seaborn.kdeplot.html
# Histogram - https://seaborn.pydata.org/generated/seaborn.distplot.html?highlight=distplot#seaborn.distplot
# countplot (Categoric or Numeric)- https://seaborn.pydata.org/generated/seaborn.countplot.html
# Barplot (Categorical & Numeric - estimate of central tendency for a numeric variable) - https://seaborn.pydata.org/generated/seaborn.barplot.html
# Boxplot - https://seaborn.pydata.org/generated/seaborn.boxplot.html

# Histogram & Kde(Kernel density estimate)
import seaborn as sns
# This produces bars and density line
sns.distplot(Df.NumVar.dropna())
# Specify Bins
sns.distplot(Df.NumVar.dropna(), bins = 100)
# Only Bins and no density line
sns.distplot(Df.NumVar.dropna(), bins = 100,kde=False)
# Only density line - Kernel Density estimation
sns.kdeplot(Df.NumVar.dropna().dropna())
sns.distplot(Df.NumVar.dropna(), bins = 100,hist=False)
# limit kde to data range
sns.kdeplot(Df.NumVar.dropna().dropna(),cut=0)

# Countplot of a single variable
sns.countplot(x="Var1", data=Df, color="c")
# Count plot of a variable within a group variable - For each unique value in Var, give the count of each unique value in Var1
sns.countplot(x="Var", hue="Var1", data=Df)
# Barplot - estimate of central tendency for a numeric variable
sns.barplot(x="Var", y="Var1", data=tips)

# Line Graph - Var(Cat), Var1(Num) - For each unique value of Var, values of Var1 are plotted as a line graph
Df.groupby(['Var'])['Var1'].plot(legend=True)

# Bar Graph - Within unique values of Var(Cat) - give sum of values in Var1(Numeric)
Df.groupby('Var').Var1.sum().plot.bar()
# 2 groupbys
Df.groupby(['Var1','Var2']).Var.sum().plot.bar()



################# Data Manipulation #################
# Drop columns
Df = Df.drop(['Var1', 'Var2'],axis=1)

### Convert data type of a var
# String/Character/object
Df['Var1'] = Df['Var1'].astype('str')
# Numeric
Df['Var1'] = Df['Var1'].astype('int')
# DateTime
Df['Var1'] =  pd.to_datetime(Df['Var1'], format='%Y-%m-%d %H:%M:%S')

### Assign values to a column or change values of a particular variable
# Assign a value
Df = DF.assign(Var="New_value")
# Create a new Var or modify existing var to assign a value (or) a formula
Df = DF.assign(Var=formula)
# Assign a Series as a new column of data frame
Df = DF.assign(NewVar=newSeries)
# Assign IDs to each row - n rows
Df = DF.assign(ID=pd.Series(range(1,n)))


### Missing data representation - np.NAN, pd.NaT
https://pandas.pydata.org/pandas-docs/stable/missing_data.html
Missing data of objects - None or NaN (NaN and None are used interchangebly)
Missing data of Numeric data - Alwayz NaN
Missing data of DateTime data - Alwayz NaT
# Number of missing values in each column of a dataframe
Df.info()
# Number of missing values of a single column
Df['Var'].isnull().sum()
# fill all NaN values with 0 - SIngle variable in a data frame
df['Var'] = df['Var'].fillna(0)
# fill entire dataset of missing values with 0
df = df.fillna(0)
# fill gaps formward and backward - Carry forward
# If there is a missing - Previous value is carry forwarded
# pad / ffill	Fill - values forward
df.fillna(method='pad')
# To limit the amount of filling NAs
df.fillna(method='pad', limit=1)
# bfill / backfill - Fill values backward
df.fillna(method='bfill')

# Cumulative sum of a column in df
df['cum_sum'] = df.val1.cumsum()

# Cumulative % of a column in df
df['cum_perc'] = 100*df.cum_sum/df.val1.sum()






















