IDEs - pyCharm, Spyder

# Install Packages in Rodeo
import pip
pip.main(['install','SciPy'])
pip.main(['install', '--upgrade', 'statsmodels'])
from scipy import *
import scipy as sy

import pip
pip.main(['install','scikit-learn'])
pip.main(['install', '--upgrade', 'statsmodels'])
from sklearn import *
import sklearn as sklearn

########################### WEB LINKS #############################
### Pandas - http://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
### Pandas - https://www.dataquest.io/blog/large_files/pandas-cheat-sheet.pdf
### Basic Operators - https://www.tutorialspoint.com/python/python_basic_operators.htm
### Datetiem - http://strftime.org/
### SQL - https://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html
### Joins - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.merge.html

# Comment or Undo comments
Ctrl+/

# Display in Terminal/Console
pd.options.display.max_rows = 1500

### Connect SQL Server to import files directly ###
import pip
pip.main(['install','--upgrade','pyodbc'])
import pyodbc 
cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=vigi-dw01;" #Change
                      "Database=Datawarehouse;" #Change
                      "Trusted_Connection=yes;")
SQLCommand = ('select * from [DataWarehouse].[dbo].[DimDate]') # SQLQuery
CalendarDays = pd.read_sql_query(SQLCommand, cnxn)   # Returns data frame                   

### Import CSV File ####
import datetime
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
Vitals = pd.read_csv("C:/Users/spashikanti/Desktop/Tri-City Sepsis #43206/Raw Data - Export from SQL/March 5 2018/Vitals_4536178.csv",
                      dtype = {'PatientID': str}, #Change data type of a variable(can be used only for string or int)
                      parse_dates = ['CollectionDateTime'], # To import variable in datetime format default format is '%d%b%Y:%H:%M:%S.%f'
                      usecols = ['VigiClientID',
                                 'PatientID',
                                 'LabKey',
                                 'LabName',
                                 'LabValue',
                                 'CollectionDateTime']) # Import only specific columns, if u wanna drop a column don't include in the list)
# Convert data type of PatientID to object
Df['PatientID'] = Df['PatientID'].astype('str')
Df['PatientID'] = Df['PatientID'].map('str')

# Data Types
Vitals.dtypes # data type/class of vars - sapply(df,class)
# Var names
list(Vitals) #Column names - names(df)

## Data type of a particular variable of a data frame
dataframe['Variable'].dtype
paq['CultureInpatientSeen'].dtype

# Convert data type to character
paq['AntiBiotic'].astype('str')
# Convert data type to u10

# Convert data type to integer/numeric
paq['AntiBiotic'].astype('int')

# Convert str/object/char to datetime var
# If a datetime var is imported as string - can be changed to datetime format using below code (Character to datetime)
im
# To convert datetime var or int variable to string/character format
Vitals['CollectionDateTime'] = Vitals['CollectionDateTime'].astype('str')

# Top few rows of a dataframe or variable
dataframe.head()
dataframe['variable'].head()


# Bottom few rows of a dataframe or a variable
paq.tail()
paq['infxCultureAdmitSeen'].tail()

### Import text file - pipe delimited ###
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_table.html
paq = pd.read_table("C:/Users/spashikanti/Desktop/Tri-City Sepsis #43206/Raw Data - Export from SQL/March 5 2018/paqSepsis_4536178.txt", 
                    dtype = {'PatientID': str},
                    sep='\t')
# http://pandas.pydata.org/Pandas_Cheat_Sheet.pdf
# https://www.dataquest.io/blog/large_files/pandas-cheat-sheet.pdf

# subset rows - delete last row
Vitals = Vitals.iloc[0:1366186]

# Subset rows - first row is 0, first column is 0
paq =paq[0:9735]
# Subset columns
paq.dtypes
x = paq.iloc[:,[0]]
# Subset column - remove single column
del df['Var']
# Remove multiple columns -?


## Basic operators in python - https://www.tutorialspoint.com/python/python_basic_operators.htm
# Filter out 1 patient
paq = paq[paq.PatientID!=6002390812]
# Filter adult patients
paq = paq[paq.age>18]

### Save python data frame/Export
paq.to_pickle("C:/Users/spashikanti/Desktop/Objectives/Python/Sepsis/DataSets/paq.pkl")
### Load python data frame/Import
paq = pd.read_pickle("C:/Users/spashikanti/Desktop/Objectives/Python/Sepsis/DataSets/paq.pkl")

# Extract Specific columns - subset
paq_ID = paq[['PatientID','VigiClientID','Gender']]

### Set Operations ###
# Both variables should be of same data type
# x UNION y
x.union(y)
# x INTERSECTION y
x.intersection(y)
# x members not in y
x.difference(y)
# y members not in x
y.difference(x)

## Difference of two sets or lists
x = set(pd.unique(df['Var']))
y = set(pd.unique(df['Var1']))
z = x-y (or) x.difference(y)


## Find observations that are common in 2 vars of a data frame
Common = set(Vitals.PatientID) & set(paq.PatientID)
# Length of a object/vector
len(Common)
# This alos gives the same result as above code
Common = set(Vitals['PatientID']).intersection(paq['PatientID'])

# Missing
https://pandas.pydata.org/pandas-docs/stable/missing_data.html

### Missing data - np.NAN = numpy.NAN, pd.NaT for DateTimes
Missing data of objects - None or NaN (NaN and None are used interchangebly)
Missing data of Numeric data - Alwayz NaN
Missing data of DateTime data - Alwayz NaT - pd.NaT

# Number of missing values in each column
df.isnull().sum()

# fill all NaN values with 0
df['Var'] = df['Var'].fillna(0)
# fill entire dataset of missing with 0
df = df.fillna(0)

# fill gaps formward and backward - Carry forward
# pad / ffill	Fill - values forward
# bfill / backfill - Fill values backward
# If there is a missing - Previous value is carry forwarded
df.fillna(method='pad')
# To limit the amount of filling NAs
df.fillna(method='pad', limit=1)

# Filter - Extract only data of Common patients in paq
# Common is a list
paq = paq[paq.PatientID.isin(list(Common))]


# Filter based on group of values
New_df = df[df.var.isin([20,22,24,25,26,28,30,36])]

# Unique Values of a variable
UqPatsVitals = pd.unique(Vitals.PatientID)
# Unique values of a variable that contains lists in each row
Df['UniqueValues'] = Df["List_Var"].apply(lambda x: np.unique(x))

# Unique labs
UqLabs = pd.unique(Vitals.LabName)

# New Variable - ifelse - where
Vitals['Temp'] =  np.where(Vitals['LabName']=='Temperature (C)',(Vitals['LabValue']*1.8)+32,np.NAN)
df['Rating'] = [1 if rating > 3 else 0 for rating in df['Rating']]

# Remove Outliers (Not is represented by ~)
Vitals = Vitals[~((Vitals['LabName']=='Creatinine Serum') & ((Vitals['LabValue']<0.1) | (Vitals['LabValue']>20)))]

Cr = Vitals[(Vitals['LabName']=='Creatinine Serum')]
max(Cr.LabValue)
min(Cr.LabValue)

LOS_paq = paq[['PatientID','admissionDate']]  
LOS_Vitals = Vitals[['PatientID','CollectionDateTime']]

## Sorting - https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
# Sort data
LOS_Vitals = LOS_Vitals.sort_values(by=['PatientID','CollectionDateTime'])
# Sort or Order by Descending order
df.sort_values('mpg',ascending=False)
# Sort few variables by ascending and others by descending - default is Ascending order
Df = Df.sort_values(by=['var1','var2','var3'], ascending=[True, False, True])


## Extracting rows within a group - https://stackoverflow.com/questions/20067636/pandas-dataframe-get-first-row-of-each-group
# Extract first observation within each PatientID - Min collection date time whole observation from Vitals within each patient
LOS_Vitals = LOS_Vitals.groupby('PatientID').first().reset_index()

## For first 2 values/rows in a group (First 2 rows)
df.groupby('id').head(2).reset_index(drop=True)
# First 5 rows
df.groupby('id').head(5).reset_index(drop=True)

## For last 2 values/rows in a group (Last 2 )
df.groupby('id').tail(2).reset_index(drop=True)
# Last 5 rows
df.groupby('id').tail(5).reset_index(drop=True)

### Remove first observation within a ID - Retain all other observations but not the first obs - if a ID is presnt only once, that 
### ID will be removed
Df = Df[Df.duplicated(['ID'], keep='first')]

### Remove last observation within a ID - Retain all other observations but not the last obs - if a ID is presnt only once, that 
### ID will be removed
Df = Df[Df.duplicated(['ID'], keep='last')]

# Left Join
LOS_Anly = pd.merge(LOS_paq, LOS_Vitals, how='left', on='PatientID')
# Joins
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


# Convert str to datetime
import datetime
LOS_Anly['admissionDate'] =  pd.to_datetime(LOS_Anly['admissionDate'], format='%Y-%m-%d %H:%M:%S.%f')

## min of two date time is taken
LOS_Anly['Arr'] = np.minimum(LOS_Anly['admissionDate'],LOS_Anly['CollectionDateTime'])
# Below code can also be used

LOS_Anly['ArrivalTime'] = np.where(LOS_Anly['admissionDate']<=LOS_Anly['CollectionDateTime'],LOS_Anly['admissionDate'],LOS_Anly['CollectionDateTime'])
LOS_Anly.dtypes

#### Difference between 2 datetimes ####
import datetime
# Convert character to datetime
paq['dischargeDate'] =  pd.to_datetime(paq['dischargeDate'], format='%Y-%m-%d %H:%M:%S.%f')
paq['admissionDate'] =  pd.to_datetime(paq['admissionDate'], format='%Y-%m-%d %H:%M:%S.%f')

# Difference between 2 datetimes
paq['LOS'] = (paq.dischargeDate-paq.admissionDate)

## Overall Difference in Days, hours, seconds, mins
LOS_Anly['LOS'] = (LOS_Anly.CollectionDateTime-LOS_Anly.admissionDate)
## Difference In days
LOS_Anly['LOS'] = (LOS_Anly.CollectionDateTime-LOS_Anly.admissionDate).dt.days
## Difference In Seconds
LOS_Anly['LOS'] = (LOS_Anly.CollectionDateTime-LOS_Anly.admissionDate).dt.seconds
## Difference In Minutes
LOS_Anly['LOS'] = ((LOS_Anly['LOS'].dt.days) * 24 * 60) + ((LOS_Anly['LOS'].dt.seconds) / 60)
# Difference In hours
LOS_Anly['LOS'] = (((LOS_Anly['LOS'].dt.days) * 24 * 60) + ((LOS_Anly['LOS'].dt.seconds) / 60))/60

### Get Year, month, date, day, week_day, day of week, day of year etc
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
### Add days or time to datetime variables
# Add 1 day
df['Var1'] = (df['Var']) + pd.Timedelta(days=1)
# Add hours
df['Var1'] = (df['Var']) + pd.Timedelta(hours=1)
# Add minutes
df['Var1'] = (df['Var']) + pd.Timedelta(minutes=60)
# Add minutes
df['Var1'] = (df['Var']) + pd.Timedelta(seconds=60)

## Add hours variable to DateTime variable - Create New DateTime var
# HoursVar is a float or int
Df['New_DateTimeVar'] = (Df['DateTimeVar']) + pd.to_timedelta(Df.HoursVar, unit='h')


### Freq Distributions & cross-tabs ###
## Unique values within a var
pd.unique(paq.AntiBio)
## Freq distribution of single variable
pd.value_counts(paq['AntiBio'].values)
## Freq dist of multiple vars - Crosstab
pd.crosstab(paq['VigiClientID'],paq['AntiBio'])

## pd.value_counts for multiple variables
Df.apply(pd.Series.value_counts)

# Number of unique values within Categorical data
for k, v in Df.nunique().to_dict().items():
    print('{}={}'.format(k,v))

### Freq table as dataframe ###
# size() is for count - outputs patientID, count(PatientID)
y = pd.DataFrame(df.groupby('PatientID').size()).reset_index()

# Count of unique values with in a list - Produces dictionary
from collections import Counter
Counter(x)
# Count of unique values where rows consits of values as List
Df["Var"].apply(lambda x: Counter(x))

#### Minimum or Maximum of 2 or more datetime variables - np.minimum / np.maximum is only for 2 vars - similar to union in r
New_Var = np.minimum(np.minimum(np.minimum(Var1,Var2),Var3),Var4)
New_Var = np.maximum(np.maximum(np.maximum(Var1,Var2),Var3),Var4)

### Execute SQL code in Python 
import pip
pip.main(['install','pandasql'])
from pandasql import *
import pandasql as ps
# Distinct values
Distinct_Vitals = Vitals[['PatientID','LabName']].drop_duplicates()
query = """select PatientID,count(PatientID) as Dist_Labs from Distinct_Vitals group by PatientID"""
Vitals_Freq = ps.sqldf(query, locals())

## Rename column names
New_df = df.rename(columns={"old_Col_name":"New_Col_name","old_Col_name1":"New_Col_name1","old_Col_name2":"New_Col_name2"})
# Rename column names by sequence order
df.columns = ['labels', 'data']

## Quantiles
# 10 reprents 10 quantiles
pd.qcut(df.var,10,labels=False,retbins=True)
# Percentiles or Quantiles
np.percentile(Df.Var, 50)
Df.Var.quantile([0.1,0.2,0.3,0.9])
# If there are missing values in data - but get percentiles
np.nanpercentile(Df.Var, 50)


## Shuffle rows of a dataframe
df.sample(frac=1)
# If you wish to shuffle your dataframe in-place and reset the index
df = df.sample(frac=1).reset_index(drop=True)

## Concatenate
Df['Var'] = Df['Var1'].astype(str) + " " +  Df['Var2'].astype(str) #Convert numeric to string for concatenation
Df['Var'] = (Df['Var1'].astype(str) + " " +  Df['Var2'].astype(str)).astype(str)

## Export
Df.to_csv('path/Df.csv')
# Export to csv - Drop rownumbers
Df.to_csv('path/Df.csv', index=False)


# Split a string column based on space,etc -Orginal column also remains in addition to new columns being added
# n correspons to number of splits - if we mention three vars, then n=2 and so on
Df[['NewVar1','NewVar2']] = Df['VarToBeSplit'].str.split(' ', n=1, expand=True)

# Assign values particular variable based on position of rows
# Should use positions for both rows and column when using iloc
Df.iloc[0:4000,columnIndex] = Value1
Df.iloc[4001:8000,columnIndex] = Value2

# Similar to rbind - concatenate/join datasets by rows
x = pd.concat([x_1,x_2,x_3,x_4])

# Compare two data frames to check for common rows
x = pd.merge(A, B, on=ColumnsList, how='inner') 

### Save all dataframe of a python/Rodeo/IDE environment
import pip
pip.main(['install','dill'])
import dill
from dill import *

# Save/dump environment/session data
filename = 'C:/Users/spashikanti/Desktop/Sepsis August 2018/Py DataFrames/globalsave.pkl'
dill.dump_session(filename)

# Load back environment/Session data - It loads back the dataframe but the dataframe names are not visible in environment --??
dill.load_session(filename)

### Save sets/objects
# Pickle.dump - save any alue first and then save the set
pickle.dump(1,open('C:/Users/spashikanti/Desktop/Sepsis August 2018/Py DataFrames/filename.obj', 'wb'))
pickle.dump(Set,open('C:/Users/spashikanti/Desktop/Sepsis August 2018/Py DataFrames/filename.obj', 'wb'))

# Load sets/object 
NewSet = pickle.load(open('C:/Users/spashikanti/Desktop/Sepsis August 2018/Py DataFrames/filename.obj', 'rb'))

# Assign value to a variable, change value of a particular variable
Df = DF.assign(Var="New_value")
# Create a new Var to assign a value (or) a formula
Df = DF.assign(Var=formula)
Df = DF.assign(NewVar=formula)
# Assign a Series as a new column of data frame
Df = DF.assign(NewVar=newSeries)



### Convert from long to wide format
# The columns used for pivoting - distinct rows for the columns selected
New_df = df.pivot(index=['Var_ID1','Var_ID2','Var_ID3'],columns='Var1', values='Var2')
# index variables are not converted to columns
# Convert all indexes to columns - run the code as is, dont assign it to a new df
df.reset_index(inplace=True)
# Manually convert index to columns - It might not work when there are multiple indexes in df
df['Var_ID1'] = df.index


# https://stackoverflow.com/questions/23626009/generate-random-numbers-replicating-arbitrary-distribution
### Replicate distribution of 1 variable to another variable
# list/set/variable of a variable
# Generating random values between 0 & 48, gives an array size of 500
x = np.random.uniform(0,48,size=500)

## replicate distribution in (x = np.random.uniform(0,48,size=500)) to a new variable of size 1000 - this size might differ from original var size
# Generate random number between 0 & 1, gives an array of size 1000
u = np.random.uniform(0,1,size=1000)
# Produces an array of 1000, that replicates distribution of x variable
new_x = np.percentile(x,(100*u).tolist())
# Convert Array to Series
new_x = pd.Series(new_x)

## Generate a random number between two values (Var1 and Var2)
# np.random.randint(low,high) - Generates integer between [low,high)
# np.random.uniform(low,high) - Generates floating value between [low,high)
# values should be mentioned as x.var1 and not Df.var1 even though var1 is a variable in Df
Df['New_Var'] = Df.apply(lambda x: np.random.uniform(x.var1, x.Var2), axis=1)
# Any random value between 10 & Var2
Df['New_Var'] = Df.apply(lambda x: np.random.uniform(10, x.Var2), axis=1)


# Add Prefix to column/Variable names
df.add_prefix('Prefix_')

# Add Suffix to column/Variable names
df.add_suffix('_Suffix')

# Replace a particular value within a variable to a new value
Df.Var[Df.Var == 'old_value'] = "New_value"


