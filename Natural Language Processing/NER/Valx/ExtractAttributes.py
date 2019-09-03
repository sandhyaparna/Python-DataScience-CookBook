#!/usr/bin/env python
# coding: utf-8

# In[33]:


pd.set_option('display.max_colwidth', -1)


# In[1]:


# https://github.com/akoumjian/datefinder
import sys 
get_ipython().system('{sys.executable} -m pip install datefinder ')


# In[1]:


import pandas as pd


# In[2]:


pd.set_option('display.max_rows', 500)
pd.options.mode.chained_assignment = None 


# In[3]:


ClinicalNotes = pd.read_excel('Craig_2019Aug7th.xlsx')

# convert text to lowercase
ClinicalNotes["Text"] = ClinicalNotes["Text"].str.lower()

# Convert index to column
ClinicalNotes['Row'] = ClinicalNotes.index

# Remove records with source type = ADT, Lab & Order
ClinicalNotes = ClinicalNotes[ClinicalNotes.SourceType.isin(["User Entered","Documentation"])]


# In[4]:


# Remove records based on number of words and if there is a number present or not and based on length of sentence
# For eg: Bmi>40 is a single word has a number but length is less
# This example has number of words=1 but length of sentence is long "wprnbwygkmfydj3ijapkf7y15yopqc0lgey2a0iryfkxbvs1zeyjm5c\r\n0vt1iwgfwijef62vbd2j+9ofnokh+moppxxdv7tzunjjcr6/peo0fticiqsb4fli7wudgfonhzab\r\nixs3biz"

# Number of words in a sentence
# df['totalwords'] = df['col'].str.split().str.len() #if /r or /n is present it is considered as space 
# df['totalwords'] = df['col'].str.count(' ') + 1 #if space is present or not
# df['totalwords'] = [len(x.split()) for x in df['col'].tolist()] #
ClinicalNotes['TotalWords'] = ClinicalNotes['Text'].str.count(' ') + 1

# Find the length of the sentence
ClinicalNotes['TextLength'] = ClinicalNotes['Text'].str.len() 

# Check if the Text has a Numeric in it or not
ClinicalNotes['NumericInText'] = ClinicalNotes['Text'].str.contains(("\d"), regex=True)


# In[5]:


ClinicalNotes = ClinicalNotes.sort_values(by=['TotalWords','TextLength','NumericInText'], ascending=[True, True, True])


# In[20]:


# Remove rows based on the conditions
Removed_ClinicalNotes = ClinicalNotes[(((ClinicalNotes.TotalWords<=5) & (ClinicalNotes.TextLength>500)) | (ClinicalNotes.NumericInText==False))]


# In[24]:


## Remove 
# TotalWords<=3 & TextLength>300
# NumericInText is not present
ClinicalNotes = ClinicalNotes[~(((ClinicalNotes.TotalWords<=3) & (ClinicalNotes.TextLength>500)) | (ClinicalNotes.NumericInText==False))]


# In[21]:


Removed_ClinicalNotes.shape


# In[40]:


# Export data
ClinicalNotes.to_excel('Py DataFrames/ClinicalNotes.xlsx',index=False)
ClinicalNotes.to_pickle("Py DataFrames/ClinicalNotes.pkl")


# In[26]:


ClinicalNotes.shape


# In[34]:





# In[162]:


Check_head = ClinicalNotes.head(50)


# In[38]:


Check_head


# In[135]:


# Identifies only single date present
# Check_head['Date'] = Check_head['Text'].str.extract(r'(\d+/\d+)', expand=True)

# Identifies only if 2 dates are present
# Check_head['Date'] = Check_head.Text.str.extract(r'(\d+/\d+)(\D+)(\d+/\d+)', expand=True)


# In[154]:


import re

# Find dates in m/d/y format
Check_head['Month_Date_Year'] = Check_head["Text"].str.findall(r'\d+/\d+/\d+')

# Find dates in m/d format
Check_head['Month_date'] = Check_head["Text"].str.findall(r'\d+/\d+')


# In[156]:


Check_head['Dates']


# In[157]:


ClinicalNotes.describe()


# In[155]:


chc = Check_head[Check_head.Row==34488]
chc


# In[213]:


# if Text is in such a way that "Any text "\n" Value" then replace \n with space --- This needs to be done after removing dates


# In[249]:


# Split datasets and once corrected attach back - 
# One efficient way might be to search for the string and extract only those that matches that string  

### Explode the Text column into multiple rows
# We start with creating a new dataframe from the series with EmployeeId as the index
ClinicalNotes_Explode = pd.DataFrame(ClinicalNotes.Text.str.split('\r\n\r\n').tolist(), index=ClinicalNotes.Row).stack()
# We now want to get rid of the secondary index
ClinicalNotes_Explode = ClinicalNotes_Explode.reset_index([0, 'Row'])
# The final step is to set the column names as we want them
ClinicalNotes_Explode.columns = ['Row', 'Text']

### Explode the Text column into multiple rows
# We start with creating a new dataframe from the series with EmployeeId as the index
ClinicalNotes_Explode = pd.DataFrame(ClinicalNotes_Explode.Text.str.split('\r\n').tolist(), index=ClinicalNotes_Explode.Row).stack()
# We now want to get rid of the secondary index
ClinicalNotes_Explode = ClinicalNotes_Explode.reset_index([0, 'Row'])
# The final step is to set the column names as we want them
ClinicalNotes_Explode.columns = ['Row', 'Text']

### Explode the Text column into multiple rows
# We start with creating a new dataframe from the series with EmployeeId as the index
ClinicalNotes_Explode = pd.DataFrame(ClinicalNotes_Explode.Text.str.split('\n').tolist(), index=ClinicalNotes_Explode.Row).stack()
# We now want to get rid of the secondary index
ClinicalNotes_Explode = ClinicalNotes_Explode.reset_index([0, 'Row'])
# The final step is to set the column names as we want them
ClinicalNotes_Explode.columns = ['Row', 'Text']

### Explode the Text column into multiple rows
# We start with creating a new dataframe from the series with EmployeeId as the index
ClinicalNotes_Explode = pd.DataFrame(ClinicalNotes_Explode.Text.str.split('\r').tolist(), index=ClinicalNotes_Explode.Row).stack()
# We now want to get rid of the secondary index
ClinicalNotes_Explode = ClinicalNotes_Explode.reset_index([0, 'Row'])
# The final step is to set the column names as we want them
ClinicalNotes_Explode.columns = ['Row', 'Text']

# donot split that data based on full stops as ClinicalNote_extract_candidates_numeric function will take care of it and 
# if starting line is date we will miss it
# ### Explode the Text column into multiple rows - Temp = 101.2 will be split at decimals so space to be added after the 
# # We start with creating a new dataframe from the series with EmployeeId as the index
# ClinicalNotes_Explode = pd.DataFrame(ClinicalNotes_Explode.Text.str.split('\. ').tolist(), index=ClinicalNotes_Explode.Row).stack()
# # We now want to get rid of the secondary index
# ClinicalNotes_Explode = ClinicalNotes_Explode.reset_index([0, 'Row'])
# # The final step is to set the column names as we want them
# ClinicalNotes_Explode.columns = ['Row', 'Text']


# In[250]:


ClinicalNotes_Explode.to_pickle("Py DataFrames/ClinicalNotes_Explode.pkl")


# In[235]:


ClinicalNotes_Explode


# In[247]:


# 4388, 38839
ClinicalNotes['Text'].iloc[38839]
ClinicalNotes[ClinicalNotes['Row']==28025].Text


# In[252]:


# Extract only those strings with wbc
wbc_Explode = ClinicalNotes_Explode[ClinicalNotes_Explode["Text"].str.contains("wbc")]


# In[253]:


# Shuffle rows
wbc_Explode = wbc_Explode.sample(frac=1)


# In[241]:


# Remove rows that have 


# In[254]:


wbc_Explode.to_csv('wbc_Explode.csv', index=False)
wbc_Explode.to_pickle("wbc_Explode.pkl")


# In[260]:


wbc_Explode


# In[256]:


Result = wbc_Explode[~(wbc_Explode["Text"].str.contains(("wbc \d"), regex=True))]
Result = Result[~(Result["Text"].str.contains(("wbc: \d"), regex=True))]


# In[257]:


Result


# In[258]:


Result.shape


# In[100]:


textToSearch = "date of service from 8/6/19 to 9/8/19"


# In[52]:


import re
import datetime
from datetime import *


# In[175]:


# First convert dates in format of %m/%d/%y to %m/%d
textToSearch = "date of service from 8/6 to 9/8/19"

# convert %m/%d/%y formatted texts to %m/%d
for match in re.findall(r'\d+/\d+/\d+', textToSearch):
    #convert match to new format
    datetime_object = datetime.strptime(match, "%m/%d/%y")
    dateNewFormat = datetime_object.strftime("%m/%d")
    #substitute the old date with the new
    textToSearch = re.sub(match, dateNewFormat, textToSearch)

# convert %m/%d formatted texts to %B%d
for match in re.findall(r'\d+/\d+', textToSearch):
    #convert match to new format
    datetime_object = datetime.strptime(match, "%m/%d")
    dateNewFormat = datetime_object.strftime("%B%d")
    #substitute the old date with the new
    textToSearch = re.sub(match, dateNewFormat, textToSearch)
print(textToSearch)


# In[176]:


# Function
textToSearch = "date of service from 8/6 to 9/8/19"
def ChangeDateFormat(TextwithDates):
    for match in re.findall(r'\d+/\d+/\d+', TextwithDates):
        #convert match to new format
        datetime_object = datetime.strptime(match,"%m/%d/%y")
        dateNewFormat = datetime_object.strftime("%m/%d")
        #substitute the old date with the new
        TextwithDates = re.sub(match, dateNewFormat, TextwithDates)
    return(TextwithDates)


# In[203]:


# Function to include 2 formats
def ChangeDateFormat(TextwithDates):
    # convert %m/%d/%y formatted texts to %m/%d
    for match in re.findall(r'\d+/\d+/\d+', TextwithDates):
        #convert match to new format
        datetime_object = datetime.strptime(match, "%m/%d/%y")
        dateNewFormat = datetime_object.strftime("%m/%d")
        #substitute the old date with the new
        TextwithDates = re.sub(match, dateNewFormat, TextwithDates)
        
    # convert %m/%d formatted texts to %B%d
    for match in re.findall(r'\d+/\d+', TextwithDates):
        #convert match to new format
        datetime_object = datetime.strptime(match, "%m/%d")
        dateNewFormat = datetime_object.strftime("%B%d")
        #substitute the old date with the new
        TextwithDates = re.sub(match, dateNewFormat, TextwithDates)
    return(TextwithDates)


# In[280]:


# function to include 4 digit Year conversion first
def ChangeDateFormat(TextwithDates):
    for match in re.findall(r'/\d{4}', TextwithDates):
        match = re.sub('/','' , match)
        #convert match to new format
        datetime_object = datetime.strptime(match, "%Y")
        dateNewFormat = datetime_object.strftime("%y")
        #substitute the old date with the new
        TextwithDates = re.sub(match, dateNewFormat, TextwithDates)

    # convert %m/%d/%y formatted texts to %m/%d
    for match in re.findall(r'\d+/\d+/\d+', TextwithDates):
        #convert match to new format
        datetime_object = datetime.strptime(match, "%m/%d/%y")
        dateNewFormat = datetime_object.strftime("%m/%d")
        #substitute the old date with the new
        TextwithDates = re.sub(match, dateNewFormat, TextwithDates)
        
    # convert %m/%d formatted texts to %B%d
    for match in re.findall(r'\d+/\d+', TextwithDates):
        #convert match to new format
        datetime_object = datetime.strptime(match, "%m/%d")
        dateNewFormat = datetime_object.strftime("%B%d")
        #substitute the old date with the new
        TextwithDates = re.sub(match, dateNewFormat, TextwithDates)
    return(TextwithDates)


# In[276]:


textToSearch = "date of service from 8/6 to 9/8/19 "
textToSearch = "date of service from 8/6 to 9/8/19 and 12/8/2019"
textToSearch = "date of service from 8/6 to 9/8/19 and 12/8/2019 and septemeber 6th 2019"
textToSearch = "date of service from 8/6 to 9/8/19 and 12/8/2019 and september 6th 2019 in 101.4 as 1400"
textToSearch = "date of service from 8/6 to 9/8/19 and 12/8/2019 and september 6th 2019 in 101.4 as @1400"
textToSearch = "date of service from 8/6 to 9/8/19 and 12/8/2019 and september 6th 2019 in 101.4 as 6000"
textToSearch = "scr remains stable 1.6-1.9. ply holding at 180s, ast/alt 76/51" # 


# In[278]:


# 5741, 7268
textToSearch = ClinicalNotes[ClinicalNotes.Row==7268].Text
textToSearch


# In[281]:


ChangeDateFormat(textToSearch)


# In[244]:


Check_head.apply(lambda x: ChangeDateFormat(x.Text), axis=1)


# In[282]:


ClinicalNotes.apply(lambda x: ChangeDateFormat(x.Text), axis=1)


# In[283]:


# error at index 7268

