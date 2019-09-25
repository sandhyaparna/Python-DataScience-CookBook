### Explore data
# Remove records based on Source type - some types of source types doesn't contain patient details
# How to remove rows that are not useful in the initial stage of Analysis - Based on length, Based on number of words & length of the text
# For datetime patterns - we need to convert dates of any format to %B%d
# Date formats - http://strftime.org/
# Functions -findall, sub

### Approach
# Original Data consists of Date, SourceType and Text; Text data may again contain different dates
# Only SourceTypes - Documentation & User Entered are suitable for our analysis - Analysis is done using length of the text, total numbers in the text, if a numeric is present in text or not and manual checking
# Excel cannot import ore export more than 32767 characters per cell and since the data contains line breaks (encoded \n), data is further truncated when imported into Python - This can be solved later on by using sql directly for exporting
# Data is further reduced based on TextLength, Total words in it and If a numeric is present in a text or not
# Data needs to be exploded based on line breaks, fullstops etc 
# Dates procesing - 


##### DateTime Packages for Text data ####
### https://github.com/alvinwan/timefhuman - time f human

### Datefinder
import datefinder

# Option1 - doesn't create a proper column 
def date_extractor(text):
    for match in datefinder.find_dates(text):
        print(match)
Check_head['Date'] = Check_head["Text"].apply(date_extractor)
Check_head.apply(lambda x: date_extractor(x.Text), axis=1)

# Option 2 - In this case, dates keep appending - doesn't reset during start of a new row
s = []
def date_extractor(text):
    for match in datefinder.find_dates(text):
        s.append(match) 
        print(s)


# Text expressions

        
        
        
### Extract dates - findall or 






### Find one date format in file and replace with another - http://strftime.org/
textToSearch = "date of service from 8/6/19 to 9/8/19"
for match in re.findall(r'\d+/\d+/\d+', textToSearch):
    #convert match to new format
    datetime_object = datetime.strptime(match, "%m/%d/%y")
    dateNewFormat = datetime_object.strftime("%m/%d/%Y %H:%M %p")
    #substitute the old date with the new
    textToSearch = re.sub(match, dateNewFormat, textToSearch)
print(textToSearch)
# Output:
date of service from 08/06/2019 00:00 AM to 09/08/2019 00:00 AM

# First convert dates in format of %m/%d/%y to %m/%d and then %m/%d  to %B%d
textToSearch = "date of service from 8/6 to 9/8/19"
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

# Definition for 2 loops
# convert %m/%d/%y formatted texts to %m/%d
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

textToSearch = "08/05/19 17:26:00 CDT 08/05 4:05 9/8/19 and 12/8/2019 temp 101 january8th and september 6 2019 march 8th xxx jan10th xyz jan 8th  na 140 (aug 07) 143 (aug 06) 141 (aug 05) " # 
