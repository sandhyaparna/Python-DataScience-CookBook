# Jupyte Notebooks:
# https://hub.coursera-notebooks.org/user/rwqjpjrvpymujvmedamubv/notebooks/Regex%20with%20Pandas%20and%20Named%20Groups.ipynb
# https://hub.coursera-notebooks.org/user/rwqjpjrvpymujvmedamubv/notebooks/Module%202%20(Python%203).ipynb


Text = "Applied Text Mining in Python"
# Characters in a string excluding spaces
len(Text) 

# All characters in a string
list(Text)

# Meta characters - Character matches
. Wildcard, matches a single character
^ Start of a string
$ end of a string
[] matches one of the set of characters within [] - if [xyz] then choose between x or y or z
[a-z] matches one of the range of chars a to z
[^abc] matches a character that is not a or b or c
a|b matches either a or b
() scoping for operators
\ Escape char for special chars (\t,\n,\b)

# Character Symbols
\b Matches word boundary
\d Any digit
\D any non-digit
\s Any whitespace
\S Any non-whitespace
\w Alphanumeric char - [A-Za-z0-9_]
\W Non-alphanumeric 

# Char Repetitions - findall
* matches zero or more occurences
+ matches one or more occurences
? matches zero or more occurences
{n} exactly n repetitions, n>=0
{n,} atleast n repetitions
{,n} atmost n repetitions
{m,n} atleast m and atmost n repetitions

# Words 
Words_in_Text = Text.split(' ')

# If Statement
if (len(Words_in_Text)>3): 
    print (Words_in_Text)

# Words with more than 3 characters - For loop
for w in Words_in_Text: 
    if len(w)>3: 
        print (w)

# istitle function is used for finding capitalized words - Checks if first character in a word is capitalized and others are small
for w in Words_in_Text: 
    if w.istitle(): 
        print (w)
 
# Words that end with a particular letter 
for w in Words_in_Text: 
    if w.endswith('g'): 
        print (w)

# Words that start with '#'
for w in Words_in_Tweet: 
    if w.startswith('#'): 
        print (w)        
# Append all the words that satisfy above consdition as a list
list=[]
for w in Words_in_Tweet: 
    if w.startswith('#'): 
        list.append(w)
print list        

# Words that start with @[A-Za-z0-9_]+ - starts with @, followed by any alphabet(upper or loer), digit or underscore, that repaets atleast once, but any number of times
for w in Words_in_Tweet: 
    if re.search('@[A-Za-z0-9_]+',w): 
        print (w)    
        
# Unique words - Capitalization is considered
set(Words_in_Text)

# We can convert all the letters in all words to lower case to identify unique words correctly
set(Text.lower().split(' '))

# Check if a particular word is in a string
for w in Text: 
    if 'to' in Text: 
        print (w)

# Check if a particular word is in set of words
for w in Words_in_Text: 
    if 'to' in Words_in_Text: 
        print (w)

# starts with a letter
Text.startswith('Letter')

# Word Comparisions
Text.isupper()
Text.islower()
Text.istitle()
Text.isalpha()
Text.isdigit()
Text.isalnum()

# String Operations
Text.lower()
Text.upper()
Text.titlecase()
Text.split('Letter')
Text.splitlines('Letter') - Splits sentence on end of lines (/n)
'Letter or Character or sub-string'.join(Text) - Each character in Text is joined by 'Letter or Character'
Text.strip() - Removes spaces and whitespace characters from the front and end of the string
Text.rstrip() - Removes spaces and whitespace characters from the end of the string
Text.find('Letter or Character or sub-string') - the space number where it was first found
Text.replace('old char','New Char')

f.readline - read the first line
f.read

# Find oval characters within Text
re.findall(r'[aeiou]',Text)

# Non-ovals
# findall() finds *all* the matches and returns them as a list of strings, with each string representing one match
re.findall(r'[^aeiou]',Text)

# Find date expressionre.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',datestr)
1 or 2 digits followed by / or -, 1 or 2 digits followed by / or -, atleast 2 digits and at most 4 digits

### Pandas ###
# Length of each row in a column 
df['text'].str.len()

# For number of words of each row in a column
df['text'].str.split().str.len()

# To check if each row of a column contains a particular word
df['text'].str.contains('Word_of_interest')

# To check how many times a digit appears in each row of a column
df['text'].str.count(r'\d')

# findall() finds *all* the matches and returns them as a list of strings, with each string representing one match

# Time format: xy:ab
findall(r'(\d?\d):(\d\d)')

# Day of the week
r'(\w+day\b)' - starts with Alpahbet, any number of chars, ends with the word day, matches word boundary
# Extract first 3 words using a function
replace(r'(\w+day\b)', lambda x: x.groups()[0][:3])

# Extract produces new columns - U can also add group name or column name for new groups using ?P
str.str.extract(r'(\d?\d):(\d\d)') - this produces 2 new columns

# What is the difference between str.extract and str.extractall
