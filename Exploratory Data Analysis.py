### CheatSheets ###
# https://www.analyticsvidhya.com/blog/2015/06/infographic-cheat-sheet-data-exploration-python/
# http://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
# https://sgfin.github.io/files/cheatsheets/Python_cheatsheet_scikit.pdf
# https://sgfin.github.io/files/cheatsheets/Python_Keras_Cheat_Sheet.pdf
# https://sgfin.github.io/files/cheatsheets/Python_cheatsheet_numpy.pdf
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PythonForDataScience.pdf
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Pandas_Cheat_Sheet_2.pdf

# https://chrisalbon.com/#Python
# http://file.allitebooks.com/20150823/Learning%20Data%20Mining%20with%20Python.pdf

################# Miscellaneous #################
# Display more data in Terminal/Console
pd.options.display.max_rows = 1500
# Display max rows and columns or entire data in Terminal/console
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199
# speed up pandas
pip install modin[ray]
import modin.pandas as pd

### How to create a dataframe
http://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html

################# DATA LOADING / IMPORTING #################
### Preview of a file-csv/text before importing as a Df
list(open('path/file.csv'))
list(open('path/file.txt'))

### SQL Server
Server - import files/tables directly 
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
# For a multi-line sql query - \ is used at end of each line
SQLCommand = ('select * \
              from [DataWarehouse].[dbo].[DimDate] \
              where y=a')
Df = pd.read_sql_query(SQLCommand, Connection)   # Returns data frame   

# From sql table - https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PandasPythonForDataScience.pdf
pd.read_sql_table('my_table', Connection)

### CSV files 
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
Df = pd.read_csv("path/file.csv",
                 dtype = {'Var1': str}, # Change data type of a variable
                 parse_dates = ['Var2'], # To import variable in datetime format default format is '%d%b%Y:%H:%M:%S.%f'
                 usecols = ['Var1', # Import specific columns
                            'Var2',
                            'Var5',
                            'Var8'],
                na_values= ['missing', 'not available', 'NA']) #Read these words as missing values

# Code to read many csv files and combine them
# https://github.com/tcbonds/lyme-disease-classifier/blob/master/2_Parsing_Climate_CSV_Files_Into_Features.ipynb

# Specify path seperately as a variable and use it in read_csv
INPUT_DIR = '.../folder/'
Df = pd.read_csv(INPUT_DIR + 'file.csv')

DATA_FOLDER = '../readonly/final_project_data/' </br>
transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))


# Firstly, create function to optimize memory for loading the data
# Can also be used for merging data sets
# From: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            IsInt = False

            mx = props[col].max()
            mn = props[col].min()

            if not np.isfinite(props[col]).all():
                props[col].fillna(-999, inplace=True)

            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()

            if -0.01 < result < 0.01:
                IsInt = True

            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)
            else:
                props[col] = props[col].astype(np.float32)

    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024 ** 2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props

def load_csv(path):
    return reduce_mem_usage(pd.read_csv(path))
# 
train_identity = load_csv(f'{PATH}/train_identity.csv')
train_df = reduce_mem_usage(pd.merge(train_transaction, train_identity, on='TransactionID', how='left'))

# Import data as reduced file (Function)
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
# Import files
%%time
warnings.simplefilter('ignore')
files = ['../input/ieee-fraud-detection/test_identity.csv', 
         '../input/ieee-fraud-detection/test_transaction.csv',
         '../input/ieee-fraud-detection/train_identity.csv',
         '../input/ieee-fraud-detection/train_transaction.csv',
         '../input/ieee-fraud-detection/sample_submission.csv']
def load_data(file):
    return reduce_mem_usage(pd.read_csv(file))
with multiprocessing.Pool() as pool:
    test_identity, test_transaction, train_identity, train_transaction, sample_submission = pool.map(load_data, files)
   
    
# Import csv 2 column file as a dictionary
feature_set = ufile.read_csv_as_dict ('data\\numeric_features.csv', 0, 1, True) #Valx - Data
greater, greater_equal, greater_equal2, lower, lower_equal, lower_equal2, equal, between, selects, connect, features, temporal, temporal_con, error1, error2, symbols, numbers, unit_special, unit_ori, unit_ori_s, unit_exp, negation = "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
def init_features ():
    feature_set = ufile.read_csv_as_dict ('data\\numeric_features.csv', 0, 1, True)
    global greater, greater_equal, greater_equal2, lower, lower_equal, lower_equal2, equal, between, selects, connect, features, temporal, temporal_con, error1, error2, symbols, numbers, unit_special, unit_ori, unit_ori_s, unit_exp, negation
    greater, greater_equal, greater_equal2, lower, lower_equal, lower_equal2, equal, between, selects, connect, features, temporal, temporal_con, error1, error2, symbols, numbers, unit_special, unit_ori, unit_ori_s, unit_exp, negation = \
    feature_set["greater"], feature_set["greater_equal"], feature_set["greater_equal2"], feature_set["lower"], feature_set["lower_equal"], feature_set["lower_equal2"], feature_set["equal"], feature_set["between"], feature_set["selects"], feature_set["connect"], feature_set["features"], feature_set["temporal"], feature_set["temporal_con"], feature_set["error1"], feature_set["error2"], feature_set["symbols"], feature_set["numbers"], feature_set["unit_special"], feature_set["unit_ori"], feature_set["unit_ori_s"], feature_set["unit_exp"], feature_set["negation"]
    temporal = temporal + '|' + temporal.replace('|', 's|') + 's'
    unit = (unit_ori + "|" + unit_ori_s.replace("|", "s|") + "s|" + unit_ori_s + "|" + temporal)
    return ""


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

### Excel files
Df = pd.read_excel('file.xlsx')
# Read Multiple sheets
xlsx = pd.ExcelFile('file.xls')
Df = pd.read_excel(xlsx, 'Sheet1')

### Extract data from json/website
!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
    -O /tmp/sarcasm.json #Saving to: ‘/tmp/sarcasm.json’
import json
with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)
    
sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline']) #headlines are stored in sentences
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

### Scraping a webpage using read_html()
# If it returns too many tables, use the 'match' parameter to find tables that contain a particular string
matching_tables = pd.read_html(url, match='String')
len(matching_tables)

### Extract tables from PDF into a dataframe
from tubula import read_pdf
Df = read_pdf('test_pdf', pages='all')


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

### Export to Excel

### Export to SQL
pd.to_sql('myDf', Connection)

################# Descriptive Stats #################
# Speed EDA - https://speedml.com/automate-exploratory-data-analysis/

# Size of a dataframe
print('This dataset has ' + str(Df.shape[0]) + ' rows, and ' + str(Df.shape[1]) + ' columns')

# Description 
def return_desc(df): 
    return print('This dataset has ' + str(df.shape[0]) + ' rows, and ' + str(df.shape[1]) + ' columns'), print(" "), print("DATA TYPES"), print (df.dtypes), print(" "), print("TOP OBSERVATIONS"), print (display(df.head(5))) , print(" ") , print("NUMBER OF MISSING VALUES IN EACH VARIABLE"), print(df.isnull().sum()), print(" "), print("PERCENTAGE OF MISSING VALUES"),print(df.isnull().sum()/len(df)*100), print(" "),print("UNIQUE VALUES IN EACH VARIABLE"), print(df.apply(lambda x: [x.unique()])), print(" "), print("NUMBER OF UNIQUE VALUES IN EACH VARIABLE"), print(df.apply(lambda x: [len(x.unique())]))

#
def SummaryTable(df):
    print('This dataset has ' + str(df.shape[0]) + ' rows, and ' + str(df.shape[1]) + ' columns')
    print("\n","TOP FEW OBSERVATIONS:")
    print(display(df.head(5)))
    print("\n","BOTTOM FEW OBSERVATIONS:")
    print(display(df.tail(5)))
    print("\n","SUMMARY of Quantitative Data:")
    print(display(df.describe()),"\n")
    summary = pd.DataFrame(df.dtypes,columns=['DataType'])
    summary = summary.reset_index()
    summary['VariableName'] = summary['index']
    summary = summary[['VariableName','DataType']]
    summary['Missing'] = df.isnull().sum().values
    summary['MissingPercentage'] = (summary['Missing']/len(df)*100).round(2)
    summary['Uniques'] = df.nunique().values
#     summary['First Value'] = df.loc[0].values
#     summary['Second Value'] = df.loc[1].values
#     summary['Third Value'] = df.loc[2].values
#     summary['Fourth Value'] = df.loc[3].values
#     summary['Fifth Value'] = df.loc[4].values
    
#     for name in summary['VariableName'].value_counts().index:
#         summary.loc[summary['VariableName'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 
    categorical_features = df.select_dtypes(include = np.object)
    print("Frequency of Categorical Data:","\n")
    for i in categorical_features.columns:
        print(i + ":" + str(categorical_features[i].nunique()))
        print(categorical_features[i].value_counts())
        print('\n')
    
    return summary
#
  SummaryTable(X_train) 

# Eg: Predicting various disease looking at a image. Each disease is represented as a binary label.  
# Print out the number of 1s for each class/disease
for column in columns:
    print(f"The class {column} has {Df[column].sum()} samples")
  
# Cross Tab of Categorical Vars by Target Variable: Freq & Percentage
for i in categorical_features.columns:
    print(i + ":" + str(categorical_features[i].nunique()))
#     print(pd.concat([pd.crosstab(train_data[i], train_data['is_late'], margins=True),
#                    pd.crosstab(train_data[i], train_data['is_late'], normalize='index')]))
    Df = pd.concat([pd.DataFrame(pd.crosstab(train_data[i], train_data['is_late'], margins=True)),
                    pd.DataFrame(train_data[i].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'),
                   pd.DataFrame(pd.crosstab(train_data[i], train_data['is_late'], normalize='index')).mul(100).round(1).astype(str) + '%'], axis=1, sort=False)
    Df.columns = ['Target_0', 'Target_1', 'Freq','FreqPercent',"Target_Percent0","Target_Percent1"]
    
    print(Df)
    print('\n')

# Frequency table for each catergorical variable
# Cross tab of Categorical variable vs Target Variable
for i in categorical_features.columns:
    print(i + ":" + str(categorical_features[i].nunique()))
    print(categorical_features[i].value_counts(), "\n",pd.crosstab(train_data[i], train_data['is_late']))
    print('\n') 

# Identify number of unique values in each variable of a Df - Constant features - Identy using Train set and remove on Train & Test set
feat_counts = train.nunique(dropna=False)
feat_counts.sort_values()[:10]
# Remove constant features
constant_features = feats_counts.loc[feats_counts==1].index.tolist()
print (constant_features)
traintest.drop(constant_features,axis = 1,inplace=True)

# Number of Unique values in a variable
Df['Var'].value_counts().shape[0]

# identify & Remove duplicated features - Identy using Train set and remove on Train & Test set
traintest.fillna('NaN', inplace=True)
#  let's encode each feature- train_enc[col] = train[col].map(train[col].value_counts())
train_enc =  pd.DataFrame(index = train.index)
for col in tqdm_notebook(traintest.columns):
    train_enc[col] = train[col].factorize()[0]
# Identify
dup_cols = {}
for i, c1 in enumerate(tqdm_notebook(train_enc.columns)):
    for c2 in train_enc.columns[i + 1:]:
        if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]):
            dup_cols[c2] = c1
# Save them
import cPickle as pickle
pickle.dump(dup_cols, open('dup_cols.p', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
# Drop from TrainTest set
traintest.drop(dup_cols.keys(), axis = 1,inplace=True)

## Identify and look at variables with a lot of unique values
mask = (nunique.astype(float)/train.shape[0] > 0.8)
train.loc[:, mask]
# different subset of unique values
mask = (nunique.astype(float)/train.shape[0] < 0.8) & (nunique.astype(float)/train.shape[0] > 0.4)
train.loc[:25, mask]

# Rows, Columns
Df.shape

# Column names
list(Df)

# Data Types
Df.dtypes
# Data type of a single var
Df['Var1'].dtype
# List of vars by datatype
Df.columns.to_series().groupby(DF.dtypes).groups

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
# Percentage of missing values
Df.isnull().sum()/len(Df)*100

# First few rows sample - Glimpse of data
Df.head()
Df['Var1'].head()

# Last few rows - Glimpse of data
Df.tail()
Df['Var1'].tail()

# Freq Dist
pd.value_counts(Df['Var1'].values)
Df['Var1'].value_counts()

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
# Z score - keep only the ones that are within +3 to -3 standard deviations
New_Df = Df[(np.abs(stats.zscore(Df[['Num_Var1','Num_Var2','Num_Var3']])) < 3).all(axis=1)]
# Filter based on z-score of a single numeric variable
df[(np.abs(stats.zscore(df.Num_Var1)))<3]
df[np.abs(df.Num_Var1-df.Num_Var1.mean()) <= (3*df.Num_Var1.std())]
# same as above code
df[np.abs((df.Num_Var1-df.Num_Var1.mean())/(df.Num_Var1.std())) <= (3)]
# Based on quantiles - Identify the 99% quantile of a NumVar & then filter the data based on that value
df[df["col"] < (df["col"].quantile(0.99))]
# Based on modified z-score
from statsmodels.robust.scale import mad
df[np.abs( (0.6745 * (df.Num_Var1-df.Num_Var1.median()))/(mad(df['Num_Var1'], c=1)) ) <= (3)]
# 1.5*IQR
IQR = (Df['Col'].quantile(0.75))-(Df['Col'].quantile(0.25))
LowerBound = (Df['Col'].quantile(0.25)) - 1.5*IQR
UpperBound = (Df['Col'].quantile(0.75)) + 1.5*IQR


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

# Check for duplicate data
columns_without_id = [col for col in Df.columns if col!='Col_ID']
# Checking for duplicates in the data.
Df[Df.duplicated(subset = columns_without_id, keep=False)]
print('The no of duplicates in the data:',Df[Df.duplicated(subset = columns_without_id, keep=False)].shape[0])
# Remove Duplicates
Df = Df.drop_duplicates()

# List of all observations within a variable
Df['Var'].values # Gives an array


################# Visualization #################
### https://python-graph-gallery.com/

### Univariate Analysis of Numeric variables
# seaborn - https://seaborn.pydata.org/examples/index.html
# Kde - https://seaborn.pydata.org/generated/seaborn.kdeplot.html
# Histogram - https://seaborn.pydata.org/generated/seaborn.distplot.html?highlight=distplot#seaborn.distplot
# countplot (Categoric or Numeric)- https://seaborn.pydata.org/generated/seaborn.countplot.html
# Barplot (Categorical & Numeric - estimate of central tendency for a numeric variable) - https://seaborn.pydata.org/generated/seaborn.barplot.html
Data is segregated and then bar plots are applied - https://python-graph-gallery.com/barplot/
# Boxplot - https://seaborn.pydata.org/generated/seaborn.boxplot.html
# Pie chart - 

# Histogram & Kde(Kernel density estimate)
import seaborn as sns
sns.set()
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
sns.barplot(x="Var", y="Var1", data=Df)

# Line Graph - Var(Cat), Var1(Num) - For each unique value of Var, values of Var1 are plotted as a line graph
Df.groupby(['Var'])['Var1'].plot(legend=True)

# Line Graph - When each variable sum is represented as a point in the graph (By year) - Each variable is a different year
plt.figure(dpi=300)
year_list = [str(x) for x in range(2000,2018)]
plt.plot(Df.sum()[4:]) # Sum of the columns from 4th column to last column
plt.title('Reported Lyme Disease Cases from 2000 to 2017')
plt.xlabel('Year')
plt.xticks(range(18), year_list, rotation=85)
plt.ylabel('Number of Reported Cases');

# Bar Graph - Within unique values of Var(Cat) - give sum of values in Var1(Numeric)
Df.groupby('Cat_Var').Var.sum().plot.bar()
# 2 groupbys
Df.groupby(['Cat_Var1','Cat_Var2']).Var.sum().plot.bar()

# Box Plot
import seaborn as sns
sns.boxplot(x=Df["Var"])
sns.boxplot(x="Cat_Var", y="Var", data=Df)
sns.boxplot(x="Cat_Var1", y="Var", hue="Cat_Var2", data=Df, palette="Set3")

# Donut plot for Target label viz
print("Distribution of data points among output class\n")
print(pd.value_counts(y_train))
import matplotlib.pyplot as plt
%matplotlib inline
# The slices will be ordered and plotted counter-clockwise.
labels = y_train.value_counts().index #labels = train_data["is_late"].value_counts().index
sizes = y_train.value_counts().values #sizes = train_data["is_late"].value_counts().values
# colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0, 0, 0)  # explode a slice if required
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.show()  

# Bar Chart for Target label viz
print("FREQUENCY OF TARGET: \n",pd.value_counts(y_train))
plt.subplot(121)
plot_tr = sns.countplot(train_data["is_late"])
plot_tr.set_title("Fraud Transactions Distribution \n 0: Not Late | 1: Late", fontsize=12)
plot_tr.set_xlabel("Is Late?", fontsize=8)
plot_tr.set_ylabel('Count', fontsize=8)
for p in plot_tr.patches:
    height = p.get_height()
    plot_tr.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/(len(train_data))*100),
            ha="center", fontsize=10) 

################# Data Manipulation #################
# Drop columns
Df = Df.drop(['Var1', 'Var2'],axis=1)
# Drop columns based on column position
Df.drop(Df.columns[[4,5]], axis=1)

# subset rows - delete last row
Df = Df.iloc[0:1366186]

# Filter/subset/check if a column/string contains only numerics implies check if the column contains any characters within it
# Extract character observations within a column (Non-numeric)
x = x[x['ZipCode'].str.contains("[a-zA-Z]+")] 

# Cbind / Concatenate 2 dataframe
New_Df = pd.concat([Df1, Df2], axis=1)
# Cbind/ Combine 2 rows
Df['New_Var'] = Df['Var1'].astype(str) + " " +  Df['Var2'].astype(str)

## Sorting - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
# Sort data
LOS_Vitals = LOS_Vitals.sort_values(by=['PatientID','CollectionDateTime'])
# Sort or Order by Descending order
df.sort_values('mpg',ascending=False)
# Sort few variables by ascending and others by descending - default is Ascending order
Df = Df.sort_values(by=['var1','var2','var3'], ascending=[True, False, True])
# Sort data - Sort by labels along an axis
df.sort_index() 
#  Sort by the values along an axis
df.sort_values(by='Country')

### Convert data type of a var
# String/Character/object
Df['Var1'] = Df['Var1'].astype('str')
# Numeric - When a numeric variable has missing data it cannot be stored as int and should be stored as float
Df['Var1'] = Df['Var1'].astype('int')
# Numeric -  When a numeric variable has missing data it cannot be stored as int and should be stored as float
Df['Var1'] = Df['Var1'].astype('float')
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

# Create a copy of a dataframe
NewDf = Df.copy()

### Missing data representation - np.NAN, pd.NaT
# Infinity inf is represented as np.inf
# -Infinity -inf is represented as -np.inf
https://pandas.pydata.org/pandas-docs/stable/missing_data.html
Missing data of objects - None or NaN (NaN and None are used interchangebly)
Missing data of Numeric data - Alwayz NaN
Missing data of DateTime data - Alwayz NaT
# All observations within a dataframe where any of the column values within the dataframe is missing
Df[Df.isnull().any(axis=1)]
# Number of missing values per row/observaton
df.isnull().sum(axis=1)
# Number of missing values in each column of a dataframe
Df.info()
# Number of missing values of a single column
Df['Var'].isnull().sum()
# fill all NaN values with 0 - Single variable in a data frame
df['Var'] = df['Var'].fillna(0)
# fill all NaN values within a variable using its mean
df['Var'] = df['Var'].fillna(df['Var'].mean())
# Fill NaNs of a particular column with a value
Df = Df.fillna({"Var": "NewValue"})
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
# bfill / backfill - Fill values backward - for a column based on another column - Values in Var column are backfilled based on UniqueId columnm
Df['Var'] = Flagler_Output.groupby(['UniqueID'])['Var'].bfill()

# Cumulative sum of a column in df
df['Var_cum_sum'] = df.val1.cumsum()
# Cumulative sum of a column within a group
Prediction_Hrs['VarForCUMSUM'] = 1
Prediction_Hrs['VarForCUMSUM'] = Prediction_Hrs.groupby(['Group'])['VarForCUMSUM'].apply(lambda x: x.cumsum())

# Cumulative % of a column in df
df['Var_cum_perc'] = 100*df.Var_cum_sum/df.val1.sum()

### Character variables encoding
Df_X is data frame with features
# http://pbpython.com/categorical-encoding.html
# http://contrib.scikit-learn.org/categorical-encoding/
# https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159

# 1. Replace/Rename/Map Values of a variable (CAN be USED for MISSING Vars)
# a)
Char_Codes = {"Char_Var1": {"Value1": New_Vaue1, "Value2": New_Vaue2},
              "Char_Var2": {"Value1": New_Vaue1, "Value2": New_Vaue2, "Value3": New_Vaue3, "Value4": New_Vaue4 }}
Df.replace(Char_Codes, inplace=True)
# b) Can be used when there are missing values - As Manual encoding doesnt change misisng values
Df['Var'] = Df['Var'].map({'Value1':New_Vaue1, 'Value2':New_Vaue2, 'Value3':New_Vaue3})
# Replace 0 or False values in the entire dataset to missing
Df = Df.replace([0,"False"],np.NAN)


# 2. Label encoding - Using Categories (CANNOT be USED for MISSING Vars)
# a) Single variable encoding
# i) Label encoding - Datatype of variable should be converted to character
Df_X['Var'] = Df_X['Var'].astype('category')
Df_X['Var'] = Df_X['Var'].cat.codes
# ii) Label encoding - Initialize label encoder - Doesnt work if there is missing data
label_encoder = preprocessing.LabelEncoder()
Df_Var_array = label_encoder.fit_transform(Df_X['Var'])

# b) MultiColumnLabelEncoder - should be used only on categorical vars
# It encodes integer variables also - so only char variables should be mentioned
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
class MultiColumnLabelEncoder:
    
    def __init__(self, columns = None):
        self.columns = columns # list of column to encode
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        
        output = X.copy()
        
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        
        return output
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
-
CharFeatures = list(Df_X.select_dtypes(include=['object']))
Df_LabelEncoded = MultiColumnLabelEncoder(CharFeatures).fit_transform(Df_X[CharFeatures])        
    
# 3. One-hot encoding - Replace existing variable values with new encoding - (CANNOT be USED for MISSING Vars)
# a) Cannot use this when there are missing values - For eg, if there are 3 uniques in a var and a few missing values, a=10, b=30, c=5, np.NAN =5, Only 2 columns will be created Var_a, var_b i.e if (Var_a=0 & Var_b=0) represents both c cat as well as missing data  
Df_OneHotEncoded = pd.get_dummies(Df_X,drop_first=True) 
# b) LabelBinarizer is also one-hot encoding - But only single variable
Label_Binarizer = LabelBinarizer()
Df_Var = Label_Binarizer.fit_transform(Df_X['Var'])
Df_Var = pd.DataFrame(Df_Var, columns=Label_Binarizer.classes_)

# 4. BinaryEncoders using category_encoders
# First the categories are encoded as ordinal, then those integers are converted into binary code, then the digits from that binary string are split into separate columns.  This encodes the data in fewer dimensions that one-hot
# Performs consistently well for actual categorical vars
Df_y = Df['Target_Var']
Df_X is data frame with features
CharFeatures = list(Df_X.select_dtypes(include=['object']))
Df_X_BinaryEncoder = ce.BinaryEncoder(cols=CharFeatures).fit(Df_X, Df_y)
Df_X_BinaryEncoder = Bank_X_BinaryEncoder.transform(Df_X)


# Subset variables/columns based on data type
Df.select_dtypes(include=['object'])
Df.select_dtypes(exclude=['object'])

# Aggregating data on multiple columns similar to sql
f = {'Field1':'sum',
         'Field2':['max','mean'],
         'Field3':['min','mean','count'],
         'Field4':'count'}
grouped = df.groupby('mykey').agg(f)
# as a data frame
grouped = pd.DataFrame(df.groupby('mykey').agg(f)).reset_index()
# Modify column names as column names will have 2 levels if many different types of aggregations are applied on a column
grouped.columns = grouped.columns.map('_'.join)
# 2 level of the column names are joined using '_'

# Caluclate % of 1s & 0s withina a categorical column i.e within each state what is the % of 1s and what is the % of 0s
# Similar to supervised ratio
Perc = Df.groupby(['CatCOlum','BinColumn'])['BinColumn'].count().rename("count")
Perc / Perc.groupby(level=0).sum()


### Convert from long to wide format
# The columns used for pivoting - distinct rows for the columns selected
New_df = df.pivot(index=['Var_ID1','Var_ID2','Var_ID3'],columns='Var1', values='Var2')
# If there are duplicated in index columns
New_df = pd.pivot_table(Df,columns=['Var1'], values='Var2', index=['Var_ID1','Var_ID2','Var_ID3']).reset_index()
New_Df = Df.pivot_Table(values='Var2', index=['Var_ID1','Var_ID2','Var_ID3'], aggfunc = np.mean)
# index variables are not converted to columns in df.pivot
# Convert all indexes to columns - run the code as is, dont assign it to a new df
df.reset_index(inplace=True)
# Manually convert index to columns - It might not work when there are multiple indexes in df
df['Var_ID1'] = df.index
# when using 2 or more variables in values of pivot_table, Var names are multi-row column
# coalesce the multi-index into a single string index without caring about the index level order
df.columns = list(map("_".join, df.columns))
# Then you can rename columns as u wish

### Wide to long format - Columns to rows
New_df = pd.melt(Df, 
 id_vars=["Date"],
 value_vars=["Type", "Value"],
 value_name="Observations")


### long to wide format & back to long format
# Fill in missing values/Dates based on 2 columns where 1st column is date and other is char
# All missing dates between the entire tables min and max dates are populated 
# Below code is a way of pivoting and unpivoting back( long to wide format & back to long format)
New_Df = Df.set_index(['DateVar','var1']).unstack(fill_value=0).asfreq('D', fill_value=0).stack().sort_index(level=1).reset_index()
# Fill in missing values/Dates based on 3 columns where 1st column is date and other 2 columns are char
New_Df = Df.set_index(['DateVar','var1','var2]).unstack(fill_value=0).unstack(fill_value=0).asfreq('D', fill_value=0).stack().stack().sort_index(level=2).reset_index()

### Date Time  
# Convert str to datetime
import datetime
Df['DateVar'] =  pd.to_datetime(Df['DateVar'], format='%Y-%m-%d %H:%M:%S.%f')

# Create Date range
df2['Date']= pd.date_range('2000-1-1', periods=6, freq='M')
# Dates as Index
index = pd.DatetimeIndex(dates)
index = pd.date_range(datetime(2012,2,1), end, freq='BM')
                       
# Min of 2 date times
Df['Var'] = np.minimum(Df['Var1'],Df['Var2'])
                       
# Difference between 2 datetimes
Df['Var'] = (Df.EndDate-Df.StartDate)
# Overall Difference in Days, hours, seconds, mins
Df['Var'] = (Df.EndDatee-Df.StartDate)
# Difference In days
Df['Var'] = (Df.EndDate-Df.StartDate).dt.days
# Difference In Seconds
Df['Var'] = (Df.EndDate-Df.StartDate).dt.seconds
# Difference In Minutes
Df['Var'] = ((Df['Var'].dt.days) * 24 * 60) + ((Df['Var'].dt.seconds) / 60)
# Difference In hours
Df['Var'] = (((Df['Var'].dt.days) * 24 * 60) + ((Df['Var'].dt.seconds) / 60))/60

# Get Year, month, date, day, week_day, day of week, day of year etc
df.Var.dt.year
# Examples and a lot more options are there
Df.DateTimeVar.dt.date
Df.DateTimeVar.dt.year
Df.DateTimeVar.dt.month
Df.DateTimeVar.dt.day
# week starts on Monday i.e Monday=0 - Both weekday and dayofweek gives same output
Df.DateTimeVar.dt.weekday
Df.DateTimeVar.dt.dayofweek
Df.DateTimeVar.dt.dayofyear
Df.DateTimeVar.dt.days_in_month
Df.DateTimeVar.dt.weekday_name
Df.DateTimeVar.dt.weekofyear                       

https://pandas.pydata.org/pandas-docs/stable/timedeltas.html
# Add days or time to datetime variables
# Add 1 day
df['Var1'] = (df['Var']) + pd.Timedelta(days=1)
# Add hours
df['Var1'] = (df['Var']) + pd.Timedelta(hours=1)
# Add minutes
df['Var1'] = (df['Var']) + pd.Timedelta(minutes=60)
# Add minutes
df['Var1'] = (df['Var']) + pd.Timedelta(seconds=60)

# Add hours variable to DateTime variable - Create New DateTime var
# HoursVar is a float or int
Df['New_DateTimeVar'] = (Df['DateTimeVar']) + pd.to_timedelta(Df.HoursVar, unit='h')
                                             
# Get Start and end date of a week based on a date variable
# Week starts on Monday - https://stackoverflow.com/questions/27989120/get-week-start-date-monday-from-a-date-column-in-python-pandas
# Start Date of the week
df['Start'] =  df['Date'] - df['Date'].dt.weekday.astype('timedelta64[D]') #This retains time 
Df['Start'] = DF['Date'].dt.to_period('W').apply(lambda r: r.start_time) #Generates only Date
# end date of the week
df['End'] =  df['Start'] +  pd.Timedelta(days=6)                       
                       
# weekday start on Sunday - https://stackoverflow.com/questions/45458525/get-week-start-date-sunday-from-a-date-column-in-python
# 1st solution
# Start Date of the week                       
df['Start'] =  df['Date'] - ((df['Date'])+(pd.Timedelta(days=1))).dt.weekday.astype('timedelta64[D]')                     
# end date of the week
df['End'] =  df['Start'] +  pd.Timedelta(days=6)
# 2nd solution                       
df = pd.DataFrame({'Date':pd.date_range('2018-08-01', periods=20)})
a =  df['Date'] - pd.offsets.Week(weekday=6)
b =  df['Date'] + pd.offsets.Week(weekday=5)
m1 = df['Date'] != (a + pd.offsets.Week())
m2 = df['Date'] != (b - pd.offsets.Week())
# Start Date of the week
df['Start'] = df['Date'].mask(m1, a)
# end date of the week
df['End'] = df['Date'].mask(m2, b)             

## BiWeek Start Date
# week starts on monday
# Calendar date starts from 1-1-1900
                       # % symbol is division remainder/modulo
Calendar = Calendar.assign(n=1)
Calendar['cum_n'] = Calendar.n.cumsum()  
df['Mod'] = (df['cum_n']-1) % 14
df['BiWeekStartDate'] = df['CollectionDate'] - df['Mod'].astype('timedelta64[D]')                       
# End is start + pd.Timedelta(days=13)
                       
# Week starts on Sunday
df['Mod'] = df['cum_n'] % 14
df['BiWeekStartDate'] = df['CollectionDate'] - df['Mod'].astype('timedelta64[D]')                       
# End is start + pd.Timedelta(days=13)

                       
### Split/Filter/Subset a dataframe into multiple dataframes based on column values
# Using a single column values
# Option 1
New_Dict_MultipleDfs =  dict(tuple(df.groupby(df['Var1'])))  
New_Dict_MultipleDfs[Unq values in var1]  #Gives splited datasets
# Option 2                       
Unique_Vigi_Org = df.Vigi_Org.unique()   # Vigi_Org is a concated column of VigiID and Organism
df_Dict = {elem : pd.DataFrame for elem in Unique_Vigi_Org}
for key in df_Dict.keys():
    df_Dict[key] = df_Dict[:][df_Dict.Vigi_Org == key]                       
# using 2/two columns - where Var1 is numeric and Var2 is string column (string column can have spaces in the string values but we can 
# first replace spaces within strings with _)
df['String_Var'] = df['String_Var'].replace(' ', '_', regex=True) 
New_Dict_MultipleDfs =  dict(tuple(df.groupby(df['Var1'].astype(str) + '_' + df['String_Var']))) 
list(New_Dict_MultipleDfs) # will give names of multiple dataframes                       
                       
# Convert dataframe variable to a list
List_X = Df['Var'].tolist()                       
                       
# Create lagged/Shift variable within a group
Df['Lag_Var'] = Df.groupby(['Group_Var'])['Var'].shift(1) #Get previous value
Df['Lead_Var'] = Df.groupby(['Group_Var'])['Var'].shift(-1) #Get Next/Lead value                   
                       
# Indexing of arrays within array
p is 
array([[1, 2, 7],
       [3, 4, 8],
       [5, 6, 9]])       
p[:,0] #Gives first column values i.e 1,3,5
p[1] #Gives 2nd row i.e 3,4,8                       
p[1,2] #Gives 3rd column value in 2nd row i.e 8                       
                       
# Create list/array of consecutive numbers between x & y
list(range(x,y))                       
                       
# Standard Deviation within each group
# Population Std dev
df.groupby('A').agg(np.std, ddof=0)                       
# Sample Std dev                       
df.groupby('A').agg(np.std, ddof=1)                       

### Join Data frames
# Left Join
Df = pd.merge(Df1, Df2, how='left', on='Var1')                      
# Joins - Diff variable/column names
A.merge(B, left_on='lkey', right_on='rkey', how='outer')
# Join only specific columns - Since we joining based on Var1 & Var2 - those 2 variables should be part of B df in the code
df = A.merge(B[['Var1','Var2','Var5','Var8']], how='outer', on=['Var1','Var2'])
# Join more than 2 dataframes - based on specific columns
df = A.merge(B[['Var1','Var2','Var5','Var8']], how='outer', on=['Var1','Var2']).merge(C[['Var6','Var2']], how='outer', on=['Var6'])
df = A.merge(B, how='outer', on=['Var1','Var2']).merge(C, how='outer', on=['Var6'])
#  Join multiple data frames
New_DF = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(Df1,Df2, how='outer', on=['Var1','Var2']),
                                                                  Df3, how='outer',on='Var1'),
                                                         Df4, how='outer',on=['Var1','Var3']),
                                                Df5, how='outer',on='Var1'),
                                         Df6, how='outer',on='Var1'),
                              Df7, how='outer',on='Var1'),
                      Df8, how='outer',on='Var1')                    

                       
# Compare/Check if 2 columns are same or equal
Df['Var1'].equals(Df['Var2'])                       
                       
# Rescale/Normalize/Transform data different scale 0-1
max* - New Maximum   ; max - Old Maximum
min* - New Minimum  ; min - Old Minimum                     
newvalue = [ (max*-min*)/(max-min)*(value-max) ] + max*
 or
newvalue = [ (max*-min*)/(max-min)*(value-min) ] + min*
                       
# https://stackoverflow.com/questions/41577468/replace-low-frequency-categorical-values-from-pandas-dataframe-while-ignoring-na
# Replace low frequency values as Other in Categorical columns
for col in list(CategoricalColumns):
    val = X_train[col].value_counts()
    y = val[val < 100].index
    X_train[col] = X_train[col].replace({x:'other' for x in y})                       
                       
# Extract columns if they have multiple categories within them 
MultipleCategoryColumns = []
for k, v in X_train[list(CategoricalColumns)].nunique().to_dict().items():
    if v >=4:
        MultipleCategoryColumns.append(k)      
                       
# Number of unique values within Categorical data
for k, v in Df.nunique().to_dict().items():
    print('{}={}'.format(k,v))                       
                       
# Extract p value based on chisq test  for multiple columns                      
for col in list(CategoricalColumns):
    pValue = scipy.stats.chi2_contingency(pd.crosstab(Df[col],Df['TargetVar']))[1]
    print('{}={}'.format(col,pValue))

# Concatenate 2 columns
Df['New_Col'] = Df['Var1'] + Df['Var2']
                       
### Pivot Data
https://pbpython.com/pandas-pivot-table-explained.html
# Convert a column as features and get binary 0 or 1 in the values
pd.get_dummies(Df, columns=['FeatureColumn'])
                       
                       
### Functions            
# function to reverse an array                       
arr =[3,2,1]
func = lambda x: (func (x[1:]) + x[:1] if x else [])
print(func(arr))
                       
### Write functions to iterate over rows of a dataframe - create new variable
# Spelling correction example
# parser.parse(row)['result'] gives the correct spelling of row value
from gingerit.gingerit import GingerIt
parser = GingerIt()
corrected = parser.parse("Analytics Vidhya is a gret platfrm to learn data scence")['result']
                       
NewVar = []
# For each row
for row in Df['Var']:
    NewVar.append(parser.parse(row)['result'])
Df['NewVar'] = NewVar

# Diff type of iterating over rows     
import textblob
from textblob import TextBlob                       
blob = TextBlob("Analytics Vidhya is a gret platfrm to learn data scence")
corrected = blob.correct() 
                       
Df["Text_Var_Corrected"] = Df["Text_Var"].apply(lambda x: str(TextBlob(x).correct()))                  
Df.apply(lambda x: functiom(x.Text), axis=1)    
# swifter can be used before apply function for faster processing
Df.swifter.apply(lambda x: functiom(x.Text), axis=1) 
                       
                       
# Extract only the sentences if a specific mentioned word appears in that sentence
# Here sentence is split based on fullstop
from nltk import sent_tokenize,word_tokenize
searched_words = ['august','jan']
ClinicalNotes['august'] = ClinicalNotes['Text_DateTimeFormatted'].apply(lambda text: [sent for sent in sent_tokenize(text)
                                       if any(True for w in word_tokenize(sent) 
                                               if w.lower() in searched_words)])
ClinicalNotes_error = ClinicalNotes[(ClinicalNotes['august'].str.len())!=0]                       
                                             
#### Trace back the error ####
import linecache
import sys

def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)

import traceback
textToSearch = "august 2017 3/125 oneal, imogene - 86yom - escherichia coli uti with bacteremia\n\nallegy: no abx\n\npmh: chf, cad, and ckd, pad with renal stent, parkinson's disease, hx ecoli bacteremia and uti august 2017, bladder suspension, cholecystectomy\n\nsh: currently residning at morning pointe alf\n\ncxc:\n4/27 blood e coli pans\n4/27 urine e coli\n\n\nabx:\nceftriaxone 4/27-\nazithro 4/27-5/1\nvanco 4/27x1\n\n5/1 approp, continue"
try:
    print(XChangeDateTimeFormat(textToSearch))
except:
#     PrintException()
    print(traceback.print_last())                    
                       
# Shuffle rows       
Df = Df.sample(frac=1)
                       
# Remove a column from a dataframe and store it as a seperate series
New_Series = Df.pop('Var')
    
# Filter Df to include only the largest cat
* Value_counts - counts =Df.Var.value_counts()
* Largest_Cat = counts.head(3).index
* isin to filter
          
          
# Tips
Guinicorn + flask.        
          
# Similar to Excel drag down - R code looping function - YoY growth/ mom
library(dplyr) 
library(lubridate) 
Date<-seq.Date(as.Date("2018-01-01"),as.Date("2020-12-01"),"months") 
Dataset<-as.data.frame(Date)
names(Dataset)[1]<-"month"
Dataset$mom<-4
for(i in 2: nrow(Dataset))
{
  Dataset[i,]$mom<-Dataset[(i-1),]$mom*1.04
  
}
#         
          

          
  
          
          
          
          
          
          

          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
          
