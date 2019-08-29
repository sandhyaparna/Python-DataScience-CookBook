### Explore data
# Remove records based on Source type - some types of documentation doesn't contain patient details
# How to remove rows that are not useful in the initial stage of Analysis - Based on length, Based on number of words & length of the text



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

### Find one format of a date in a string and convert it to another











