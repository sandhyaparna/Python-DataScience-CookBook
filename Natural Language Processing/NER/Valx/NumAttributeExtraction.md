## Explore data
* Remove records based on Source type - some types of source types doesn't contain patient details
* How to remove rows that are not useful in the initial stage of Analysis - Based on length, Based on number of words & length of the text
* For datetime patterns - we need to convert dates of any format to %B%d
* Date formats - http://strftime.org/
* Functions - findall, sub. replace

## Approach
* Original Data consists of Date, SourceType and Text; Text data may again contain different dates
* Data Cleaning - Convert to lower case, remove special chars from start of a string, 
* Only SourceTypes - Documentation & User Entered are suitable for our analysis 
* Excel cannot import ore export more than 32767 characters per cell and since the data contains line breaks (encoded \n), data is further truncated when imported into Python - This can be solved later on by using sql directly for exporting
* Dates processing - first 
* Data is further reduced based on TextLength, Total words in it and If a numeric is present in a text or not
* Data needs to be exploded based on line breaks or special characters that are used to end a sentence and then again Dateformatting is to be done (In some observations if error is present whole obs wont be formatted) - Data should not be exploded by fullstop because a para might contain sentences for a day and if we split on fullstops, date will be lost
  * Eg: 3/1 procal 31.13 on adm. wbc normalized, remains afebrile, on nc 1l, tx to floor, approp, cont.
* Most frequent numerical attributes are chosen so that we capture language semantics of various writing ways. Value counts is used to return words with top freqs
  * WBC
  * yof/yom - yo female or male - Age
  * crCl or Scr
  * Temp - tmax, tmin, febrile, afebrile, fever, fib, afib, temperature
  * qtc
  * pulse, respiratory, gluc, nasal cannula (nc, oxygen), protein, platelet, bmi, inr, fi02
* Function to process whole file - differnt sentences within a para
* Understand Valx_core.py in depth - 



## Challenges
#### Dates
* 2/29 : Day is out of range for month - Manual correction
* Sept is written instead of Sep
* If 1 date is written in 2 different formats - 06/25 gets converted to 006/25 (Happens when 6/25 is present before 06/25); 08/21 gets converted to 0August21 - %B%d is replaced with %B%#d
* If an error happens in a sentence, the ehole sentence after the error remains unchanged - Try to trace back the error AND hence, blood pressure in the format of p/q cannot be extracted fully because of this issue
#### UserEntered vs Documentation
* A patients will have multiple records - Each additional day may be written as a new paragraph and concatenated to previous notes subsequently
#### Numerical Attributes & their values
* If multiple values are present irrespective of their relation to Numerical attribute, they also get extracted
  * Any var not part of 'variable_feature_umls' will not be rceognized - UMLS concepts needs to be used
  * 
* If value is before Numerical attribute (Tip: If only 1 numerical attribute is in that sentence, value gets extracted for it, no matter if it is before or after. But if 2 numerical attributes value should be after the numerical attribute for correct extraction) 
* Data gets split again based on fullstops etc


## Diff date patterns covered: Coverted to MonthDD format
* sep 6; sep 6th; sep 1st; sep 2nd, sep 3rd (space between month & date AND no space between month and date) & (full forms of Months)
* jan 2014  & (full forms of Months)
* jan 6 H:M:S & (full forms of Months)
* Jan 6 H:M & (full forms of Months)
* MM/DD/YYYY (H:M:S) & (H:M)
* MM/DD/YY (H:M:S) & (H:M)
* MM/DD (H:M:S) & (H:M)

### DateTime formats not covered (Exceptions)
* MM.DD
* DD/MM gets converted to MM/DD format
* Diff date formats: 2012/4/11
* If Month and Year are written as Sep 17, even when it sep 2017 it will be considered as Sep 17th
* Typos - Back to back dates 9/229/23; If 4/6 is written as 4\6; 11//23 instead of 11/23
* 5-6/10 implies 5th to 6th October (Gets converted to 5-June10

#### DateTime Packages for Text data ####
* https://github.com/alvinwan/timefhuman - time f human


## Code
* ClinicalNotes_WBC['Text_DateTimeFormatted'].str.findall(r'wbc')
* re.findall(r'\d+/\d+/\d+', textToSearch)




