### Explore data
# Remove records based on Source type - some types of documentation doesn't contain patient details
# How to remove rows that are not useful in the initial stage of Analysis - Based on length, Based on number of words & length of the text
# For datetime patterns - we need to convert dates of any format to %B%d
# Date formats - http://strftime.org/
# Functions -findall, sub

### Approach
# Data consists of Date, SourceType and Text



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

textToSearch = "08/05/19 17:26:00 CDT 08/05 4:05 9/8/19 and 12/8/2019 temp 101 january8th and september 6 2019 march 8th xxx jan10th xyz jan 8th  na 140 (aug 07) 143 (aug 06) 141 (aug 05) " # 
MonDD = r"jan\d+|feb\d+|mar\d+|apr\d+|may\d+|jun\d+|jul\d+|aug\d+|sep\d+|oct\d+|nov\d+|dec\d+"
# First convert jan to january etc
for match in re.findall(MonDD, textToSearch):
    datetime_object = datetime.strptime(match,"%b%d")
    dateNewFormat = datetime_object.strftime("%B%d ")
    #substitute the old date with the new
    textToSearch = re.sub(match, dateNewFormat, textToSearch)

Mon_DD = r"jan \d+|feb \d+|mar \d+|apr \d+|may \d+|jun \d+|jul \d+|aug \d+|sept \d+|oct \d+|nov \d+|dec \d+"
for match in re.findall(Mon_DD, textToSearch):
    datetime_object = datetime.strptime(match,"%b %d")
    dateNewFormat = datetime_object.strftime("%B%d ")
    #substitute the old date with the new
    textToSearch = re.sub(match, dateNewFormat, textToSearch)

MonthDD = r"january\d+|february\d+|march\d+|april\d+|may\d+|june\d+|july\d+|august\d+|september\d+|october\d+|november\d+|december\d+"
for match in re.findall(MonthDD, textToSearch):
    datetime_object = datetime.strptime(match,"%B%d")
    dateNewFormat = datetime_object.strftime("%B%d ")
    #substitute the old date with the new
    textToSearch = re.sub(match, dateNewFormat, textToSearch)

Month_DD = r"january \d+|february \d+|march \d+|april \d+|may \d+|june \d+|july \d+|august \d+|september \d+|october \d+|november \d+|december \d+"
for match in re.findall(Month_DD, textToSearch):
    datetime_object = datetime.strptime(match,"%B %d")
    dateNewFormat = datetime_object.strftime("%B%d ")
    #substitute the old date with the new
    textToSearch = re.sub(match, dateNewFormat, textToSearch)
print(textToSearch)

# If error then return the original text itself
def ChangeDateTimeFormat(TextwithDates):
    try:
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
        for match in re.findall(r'\d+/\d+ \d+:\d+:\d+', TextwithDates):
            datetime_object = datetime.strptime(match, "%m/%d %H:%M:%S")
            dateNewFormat = datetime_object.strftime("%B%d_%H:%M:%S")
            #substitute the old date with the new
            TextwithDates = re.sub(match, dateNewFormat, TextwithDates)

        for match in re.findall(r'\d+/\d+ \d+:\d+', TextwithDates):
            datetime_object = datetime.strptime(match, "%m/%d %H:%M")
            dateNewFormat = datetime_object.strftime("%B%d_%H:%M")
            #substitute the old date with the new
            TextwithDates = re.sub(match, dateNewFormat, TextwithDates)

        for match in re.findall(r'\d+/\d+', TextwithDates):
            datetime_object = datetime.strptime(match, "%m/%d")
            dateNewFormat = datetime_object.strftime("%B%d")
            #substitute the old date with the new
            TextwithDates = re.sub(match, dateNewFormat, TextwithDates)

        MonDD = r"jan\d+|feb\d+|mar\d+|apr\d+|may\d+|jun\d+|jul\d+|aug\d+|sep\d+|oct\d+|nov\d+|dec\d+"
        # First convert jan to january etc
        for match in re.findall(MonDD, TextwithDates):
            datetime_object = datetime.strptime(match,"%b%d")
            dateNewFormat = datetime_object.strftime("%B%d ")
            #substitute the old date with the new
            TextwithDates = re.sub(match, dateNewFormat, TextwithDates)
        Mon_DD = r"jan \d+|feb \d+|mar \d+|apr \d+|may \d+|jun \d+|jul \d+|aug \d+|sept \d+|oct \d+|nov \d+|dec \d+"
        for match in re.findall(Mon_DD, TextwithDates):
            datetime_object = datetime.strptime(match,"%b %d")
            dateNewFormat = datetime_object.strftime("%B%d ")
            #substitute the old date with the new
            TextwithDates = re.sub(match, dateNewFormat, TextwithDates)

        MonthDD = r"january\d+|february\d+|march\d+|april\d+|may\d+|june\d+|july\d+|august\d+|september\d+|october\d+|november\d+|december\d+"
        for match in re.findall(MonthDD, TextwithDates):
            datetime_object = datetime.strptime(match,"%B%d")
            dateNewFormat = datetime_object.strftime("%B%d ")
            #substitute the old date with the new
            TextwithDates = re.sub(match, dateNewFormat, TextwithDates)

        Month_DD = r"january \d+|february \d+|march \d+|april \d+|may \d+|june \d+|july \d+|august \d+|september \d+|october \d+|november \d+|december \d+"
        for match in re.findall(Month_DD, TextwithDates):
            datetime_object = datetime.strptime(match,"%B %d")
            dateNewFormat = datetime_object.strftime("%B%d ")
            #substitute the old date with the new
            TextwithDates = re.sub(match, dateNewFormat, TextwithDates)
        return(TextwithDates)
    except Exception:
        return(TextwithDates)









