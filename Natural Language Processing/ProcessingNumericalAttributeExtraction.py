### Explore data
# Remove records based on Source type - some types of documentation doesn't contain patient details
# How to remove rows that are not useful in the initial stage of Analysis - Based on length, Based on number of words & length of the text

# Date formats - http://strftime.org/

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













