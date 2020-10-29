# Python 3 Text Processing with NLTK Cookbook
# https://ucilnica.fri.uni-lj.si/pluginfile.php/46018/mod_resource/content/1/Python%203%20Text%20Processing%20with%20NLTK%203%20Cookbook.pdf
# https://www.analyticsvidhya.com/blog/2015/06/regular-expression-python/ - Regular Expressions
# https://www.analyticsvidhya.com/blog/2014/11/text-data-cleaning-steps-python/ - Data Cleaning

import nltk
from nltk import *

### Import sample text
https://gist.github.com/kunalj101/ad1d9c58d338e20d09ff26bcc06c4235
# load the dataset
data = open('data/corpus').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(content[1:])

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

# Length of the text column as a new column
Df["Name Length"]= data["Name"].str.len() 
Df["Name Length"]= data["Name"].apply(lambda x: len(str(x)))
Df["Name Length"]= data["Name"].apply(len)

# Split the string into words
Df["text"].str.split()

# Remove any leading and trailing spaces
Df["text"] = Df["text"].apply(lambda x: x.strip())

# Remove ip address
TrainData["comment_text"] = TrainData["comment_text"].str.replace(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "")
# Extract ip address in a comment and then count
df['ip']=df["comment_text"].apply(lambda x: re.findall("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",str(x)))
#count of ip addresses
df['count_ip']=df["ip"].apply(lambda x: len(x))

# Remove any links (whole link is replaced)
TrainData["comment_text"] = TrainData["comment_text"].str.replace(r"http://.*", "")
TrainData["comment_text"] = TrainData["comment_text"].str.replace(r"https://.*", "")
# Extract links
df['link']=df["comment_text"].apply(lambda x: re.findall("http://.*com",str(x)))
#count of links
df['count_links']=df["link"].apply(lambda x: len(x))

#article ids
df['article_id']=df["comment_text"].apply(lambda x: re.findall("\d:\d\d\s{0,5}$",str(x)))

#username - regex for     Match anything with [[User: ---------- ]]
# regexp = re.compile("\[\[User:(.*)\|")
df['username']=df["comment_text"].apply(lambda x: re.findall("\[\[User(.*)\|",str(x)))

# Number of the words in the string - First split the string into words and then count them
Df["text"].str.split().apply(lambda x: len(x))
df['count_word']=df["comment_text"].apply(lambda x: len(str(x).split()))

# Number of sentences '\n' can be used to count the number of sentences in each comment
df['count_sent']=df["comment_text"].apply(lambda x: len(re.findall("\n",str(x)))+1)

#Unique word count - Number of unique words - If unique words are more, that can be a sign of spamming 
df['count_unique_word']=df["comment_text"].apply(lambda x: len(set(str(x).split())))

#punctuation count - Number of punctuations
df["count_punctuations"] =df["comment_text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

#upper case words count - whole upper case written words
df["count_words_upper"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

#title case words count
df["count_words_title"] = df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

#Number of stopwords
df["count_stopwords"] = df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

#Average length of the words 
df["mean_word_len"] = df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
#Average length of the words - takes special characters into account - Length of the sentence divided by 
Df["Name Length"]/(df['count_word']+1)



### Spelling Correction 
# Option1: https://www.analyticsvidhya.com/blog/2018/02/natural-language-processing-for-beginners-using-textblob/
import textblob
from textblob import TextBlob
Df["Text_Var_Corrected"] = Df["Text_Var"].apply(lambda x: str(TextBlob(x).correct()))

# Option2: Takes a lot of time for many rows
from gingerit.gingerit import GingerIt
parser = GingerIt()
corrected = parser.parse("Analytics Vidhya is a gret platfrm to learn data scence")['result']                      
NewVar = []
# For each row
for row in Df['Var']:
    NewVar.append(parser.parse(row)['result'])
Df['NewVar'] = NewVar

# Option3: Takes more time even for a single sentence and doesn't do the correction properly
!pip install pyspellchecker
from spellchecker import SpellChecker
SpellChecker().correction("The smelt of fliwers bring back memories")

# Option4: hunspell (Didn't try)
https://github.com/hunspell/hunspell

### Regex - Pattern Extraction/Matching Findall search
http://www.pyregex.com/
Regex Cheat Sheet: https://www.rexegg.com/regex-quickstart.html
DateTime Pattern http://strftime.org/
https://www.analyticsvidhya.com/blog/2015/06/regular-expression-python/?utm_source=blog&utm_medium=learning-path-nlp-2020
    
Import re

# Findall on a text column
ClinicalNotes_WBC['wbcRows'] = ClinicalNotes_WBC['Text_DateTimeFormatted'].str.findall(r'wbc') #On entire column

textToSearch = "date of service from 8/6 to 9/8/19 "
re.findall(r'\d+/\d+/\d+', textToSearch) #textToSearch is a string

# find the non-alphanumeric characters from the string
re.findall("\W+",string)

# Difference between findall & contains - Findall looks for exact pattern
subset of Data = UserEntered_Num[UserEntered_Num['Text_DateTimeFormatted'].str.contains(WordsList)]

# re.match(pattern, string) - Find matches at the beginning of the string only
result = re.match(r'AV', 'AV Analytics Vidhya AV')
Print(result.group(0)) # Outputs 'AV' if it is present at the start of the string - 'AV Analytics Vidhya AV'
print ('\nStarting position of the match :',result.start()) # Outputs 0 
print ('Ending position of the match :',result.end()) # Outputs 2 

# re.search(pattern, string) -  It is similar to match() but it doesn’t restrict us to find matches at the beginning of the string only.
result = re.search(r'Analytics', 'AV Analytics Vidhya AV')
print result.group(0)
Output:
Analytics

# re.findall (pattern, string):It helps to get a list of all matching patterns. It has no constraints of searching from start or end. In given string it will return both occurrence of AV. 
result = re.findall(r'AV', 'AV Analytics Vidhya AV')
print result
Output:
['AV', 'AV']

# re.split(pattern, string, [maxsplit=0]): This methods helps to split string by the occurrences of given pattern.
result=re.split(r'y','Analytics')
result
Output:
['Anal', 'tics']
# Max split can be used to restrict splits
result=re.split(r'i','Analytics Vidhya',maxsplit=1)
result
Output:
['Analyt', 'cs Vidhya']

# re.sub(pattern, repl, string): It helps to search a pattern and replace with a new sub string. If the pattern is not found, string is returned unchanged.
result=re.sub(r'India','the World','AV is largest Analytics community of India')
result
Output:
'AV is largest Analytics community of the World'

# re.compile(pattern, repl, string): We can combine a regular expression pattern into pattern objects, which can be used for pattern matching. It also helps to search a pattern again without rewriting it.
import re
pattern=re.compile('AV')
result=pattern.findall('AV Analytics Vidhya AV')
print result
Output:
['AV', 'AV']
result2=pattern.findall('AV is largest analytics community of India')
print result2
Output:
['AV']

# Rule based matching using spcay 
# With this spaCy matcher, you can find words and phrases in the text using user-defined rules. While Regular Expressions use text patterns to find words and phrases, the spaCy matcher not only uses the text patterns but lexical properties of the word, such as POS tags, dependency tags, lemma, etc.
# So, the pattern is a list of token attributes. For example, ‘TEXT’ is a token attribute that means the exact text of the token. There are, in fact, many other useful token attributes in spaCy which can be used to define a variety of rules and patterns.
# https://spacy.io/usage/rule-based-matching
import spacy
nlp = spacy.load('en_core_web_sm')
# Import spaCy Matcher
from spacy.matcher import Matcher
# Initialize the matcher with the spaCy vocabulary
matcher = Matcher(nlp.vocab)
doc = nlp("Some people start their day with lemon water")
# Define rule - we define the rule/pattern for what we want to extract from the text.
# our objective in the below pattern is that whenever “lemon” is followed by the word “water”, then the matcher should be able to find this pattern in the text.
pattern = [{'TEXT': 'lemon'}, {'TEXT': 'water'}]
# Add rule
matcher.add('rule_1', None, pattern)
# Extraction matched text using rule
matches = matcher(doc)
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)
# Below pattern finds the word “book” only if it has been used in the sentence as a noun.
pattern = [{'TEXT': 'book', 'POS': 'NOUN'}]

### Noise Removal
# Tokenization - splitting text into words (Removal of stop words, etc) - https://www.analyticsvidhya.com/blog/2020/06/hugging-face-tokenizers-nlp-library/
# http://www.nltk.org/api/nltk.tokenize.html - Different types of tokenization 

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

## Stop words
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
set(stopwords.words('english'))
stop_words = set(stopwords.words('english'))  # stop_words = nltk.corpus.stopwords.words('english')
new_stop_words = ['stopWord1','stopWord2']
stop_words.extend(new_stop_words)
Df["Text_Var3"] = Df["Text_Var2"].apply(lambda x: [item for item in x if item not in stop_words]) #Removes stop words from the Text var2
# Using spacy
from spacy.lang.en import English
nlp = English()
my_doc = nlp(text)
text = """He determined to drop his litigation with the monastry"""
from spacy.lang.en.stop_words import STOP_WORDS
# Using Gensim
from gensim.parsing.preprocessing import remove_stopwords
result = remove_stopwords("""He determined to drop his litigation with the monastry""")
# Another way
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


# Word Tokenization should be performed first and the removal of stop words - As Text may have sentences and punctuations are split in tokenization

# Convert text data into small letter
Df["Text_Var1"] = Df["Text_Var"].str.lower()

# Extract Hashtagged words into a seperate column - Should not be Tokenized
Df["Text_Var5"] = Df["Text_Var"].str.findall(r'#.*?(?=\s|$)')

# Remove numbers from Text data - Should not be Tokenized
Df["Text_Var5"] = Df["Text_Var"].str.replace('\d+', '')

# Remove Hashtag and not the word associated with it, a few other expressions
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
return df

# Remove Hashtag words from text
Df["Text_Var"] = Df["Text_Var"].str.replace('#[\w]*', '')

# https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch07_Analyzing_Movie_Reviews_Sentiment/Text%20Normalization%20Demo.ipynb

# Remove/strip HTML
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

# Removing accented characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# Replace multiple texts at once
from collections import OrderedDict
# replacement Patterns
ReplacementSet = OrderedDict([("Text", "Replace_text"), ("Text1", "Replace_text1"),("won\'t", "will not"),("can\'t", "cannot"),("i\'m", "i am"),("ain\'t", "is not"),
 ("(\w+)\'ll", "\g<1> will"), ("(\w+)n\'t", "\g<1> not"), ("(\w+)\'ve", "\g<1> have"), ("(\w+)\'s", "\g<1> is"), ("(\w+)\'re", "\g<1> are"), ("(\w+)\'d", "\g<1> would")])
def replace_all(text, dic=ReplacementSet):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text
Df["Text_Var"] = Df["Text_Var"].apply(replace_all)

import replacer
# Replacement - Contracdictions
import re
replacement_patterns = [
 (r'won\'t', 'will not'),
 (r'can\'t', 'cannot'),
 (r'i\'m', 'i am'),
 (r'ain\'t', 'is not'),
 (r'(\w+)\'ll', '\g<1> will'),
 (r'(\w+)n\'t', '\g<1> not'),
 (r'(\w+)\'ve', '\g<1> have'),
 (r'(\w+)\'s', '\g<1> is'),
 (r'(\w+)\'re', '\g<1> are'),
 (r'(\w+)\'d', '\g<1> would')]
class RegexpReplacer(object):
 def __init__(self, patterns=replacement_patterns):
 self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
 def replace(self, text):
  s = text
  for (pattern, repl) in self.patterns:
   s = re.sub(pattern, repl, s)
  return s
replacer = RegexpReplacer()
Df["Text_Var"] = replacer.replace(Df["Text_Var"])

# Removing repeated chars
# Improved version for repeated chars
import re
from nltk.corpus import wordnet
class RepeatReplacer(object):
 def __init__(self):
  self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
  self.repl = r'\1\2\3'
 def replace(self, word):
  if wordnet.synsets(word):
    return word
  repl_word = self.repeat_regexp.sub(self.repl, word)
  if repl_word != word:
    return self.replace(repl_word)
  else:
    return repl_word
replacer = RepeatReplacer()
Df["Text_Var"] = replacer.replace(Df["Text_Var"])
# Old version for repeated chars
import re
class RepeatReplacer(object):
 def __init__(self):
  self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
  self.repl = r'\1\2\3'
 def replace(self, word):
  repl_word = self.repeat_regexp.sub(self.repl, word)
  if repl_word != word:
   return self.replace(repl_word)
  else:
   return repl_word


# Spelling Correction - enchant package stopped working - No update
import enchant
from nltk.metrics import edit_distance
class SpellingReplacer(object):
 def __init__(self, dict_name='en', max_dist=2):
  self.spell_dict = enchant.Dict(dict_name)
  self.max_dist = max_dist
 def replace(self, word):
  if self.spell_dict.check(word):
   return word
  suggestions = self.spell_dict.suggest(word)
  if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
   return suggestions[0]
  else:
   return word
replacer =  SpellingReplacer()
Df["Text_Var"] = replacer.replace(Df["Text_Var"])

# CustomSpellingReplacer can also be used
d = enchant.DictWithPWL('en_US', 'mywords.txt')
replacer = CustomSpellingReplacer(d)
Df["Text_Var"] = replacer.replace(Df["Text_Var"])


# Replace with Synonyms - Your CSV file should consist of two columns, where the first column is the word and the second column is the synonym meant to replace it
class WordReplacer(object):
 def __init__(self, word_map):
    self.word_map = word_map
 def replace(self, word):
    return self.word_map.get(word, word)
class CsvWordReplacer(WordReplacer):
 def __init__(self, fname):
  word_map = {}
  for line in csv.reader(open(fname)):
   word, syn = line
   word_map[word] = syn
  super(CsvWordReplacer, self).__init__(word_map)    
import csv
replacer = CsvWordReplacer('synonyms.csv')  

# Remove punctuations from strings
import string
s = '... some string with punctuation ...'
s = s.translate(None, string.punctuation)
Df["Text_Var"] = Df["Text_Var"].apply(lambda x: x.translate(None, string.punctuation))

# Remove Numbers from strings
Df["Text_Var"] = Df["Text_Var"].apply(lambda x: x.translate(None, string.digits))

# Remove Special characters
def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text
Df["comment_text"].str.replace(r'[^a-zA-Z]', " ") # replaces anything other than alphabet character, remove numbers any character

# Word Tokenization - words are seperated by commas
Df["Text_Var2"] = Df["Text_Var1"].apply(nltk.word_tokenize)   # Df["Text_Var2"] = Df["Text_Var"].str.lower().str.split() -- Same as Tokenization but . and , are attached to words as is in the text

# Removes stop words from the Text var2 i.e Tokenazied column
Df["Text_Var3"] = Df["Text_Var2"].apply(lambda x: [item for item in x if item not in stop_words])

# Remove stop words  - Raw Text - Non Tokenized column
Df["Text_Var4"] = Df["Text_Var1"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

# Removes punctuations,# as well as tokenize - Within RegexpTokenizer function any expression string can be used 
tokenizer = RegexpTokenizer(r'\w+') #Alpha-numeic
Df["Text_Var4"] = Df["Text_Var"].apply(tokenizer.tokenize)

### Expand Contractions
cList = { 
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"I'd": "I had / I would",
"I'd've": "I would have",
"I'll": "I shall / I will",
"I'll've": "I shall have / I will have",
"I'm": "I am",
"I've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
import re
c_re = re.compile(r'\b(?:%s)\b' % '|'.join(cList.keys()))
def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)
Df["Text_Var"] = Df["Text_Var"].apply(expandContractions)

# 2nd way - Using Contractions package
!pip install contractions
from contractions import contractions_dict
import re
# 
import contractions
contractions.fix('I didn\'t like this t-shirt')
contractions.fix("I don't have to be there at 9 o'clock") #Result: 'I do not have to be there at 9 of the clock'
Df["Text"].apply(lambda x: contractions.fix(x))
# 
def expand_contractions(text, contraction_mapping=contractions_dict):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
# Eg: expand_contractions("I don't have to be there at 9 o'clock") # Output: I do not have to be there at 9 of the clock


### Tokenizing - https://www.analyticsvidhya.com/blog/2020/06/hugging-face-tokenizers-nlp-library/
# Removes punctuations,# as well as tokenize - Within RegexpTokenizer function any expression string can be used 
tokenizer = RegexpTokenizer(r'\w+') #Alpha-numeic
Df["Text_Var4"] = Df["Text_Var"].apply(tokenizer.tokenize)
# Sentence Tokenization
from nltk.tokenize import sent_tokenize
Df["Text_Var2"] = Df["Text_Var1"].apply(nltk.sent_tokenize)
# Basic word tokenization
from nltk.tokenize import word_tokenize 
Df["Text_Var2"] = Df["Text_Var1"].apply(nltk.word_tokenize)
# PunktWordTokenizer - splits on punctuation, but keeps it with the word instead of creating separate tokens for punctuations
from nltk.tokenize import PunktWordTokenizer
tokenizer = PunktWordTokenizer()
Df["Text_Var"] = tokenizer.tokenize(Df["Text_Var"])
# WordPunctTokenizer - splits all punctuation into separate tokens
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
Df["Text_Var"] = tokenizer.tokenize(Df["Text_Var"])
# TreebankWordTokenizer
from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
Df["Text_Var"] = tokenizer.tokenize(Df["Text_Var"])

## Using spacy - Tokenize
# Load model to return language object
nlp = spacy.load('en')
df['Tokenized_Text'] = [nlp(text) for text in df.Text_Var]
# Sum the number of tokens in each Doc
df['num_tokens'] = [len(token) for token in df.Tokenized_Text]
# Visualize histogram of tokens per tweet
g = sns.distplot(df.num_tokens)


### De-tokenize
import re
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(untokenize)
   
 ### POS (Part of speech) tagging 
# Single Sentence
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(nltk.pos_tag)
# Sentences
Df["Text_Var"] = Df["Sent_Tokenized_Text_Var"].apply(nltk.sent_tokenize)
# Build pos tagger using Treebank data
nltk.download('treebank')
from nltk.corpus import treebank
from nltk.tag import tnt
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(treebank.tagged_sents())
# Save as pickle and then import it to use it (or) run above code and use the function
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(tnt_pos_tagger.tag)

# When POS(Part of Speech) tagging is done before Lemmetizing - POs tagging helps in a more meaningful Stemming
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
Df["Tokenized_Text_Var"] = Df["Text_Var"].apply(nltk.word_tokenize)
Df["POSTagged_Text_Var"] = Df["Tokenized_Text_Var"].apply(nltk.pos_tag)

# POS(Part of Speech) using spacy
import spacy
sp = spacy.load('en_core_web_sm')
sen = sp(u"I like to play football. I hated it in my childhood though")
# word.pos_ for coarse POS
# word.tag_ for fine-grained
# To view description of either ones spacy.explain(tag)
# POS of all words in a sentence
for word in sen:
    print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}') # 12 is number of spaces added between text and coarse POS; 10 is number of spaces between coarse POS and fine-grained POS
# POS for a single word
sen = sp(u'Can you google it?')
word = sen[2]
print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')
# POS dependancy visualization
from spacy import displacy
displacy.render(sen,style='dep', jupyter=True)

spacy.explain('nsubj')
# Dependancy parsing
import spacy
sp = spacy.load('en_core_web_sm')
sen = sp(u"I like to play football. I hated it in my childhood though")
# token.dep_ gives dependency tag
# The dependency tag ROOT denotes the main verb or action in the sentence. The other words are directly or indirectly connected to the ROOT word of the sentence.
for token in sen:
    print(token.text, "-->", token.dep_)

# Noun Phrasing
import spacy
sp = spacy.load('en_core_web_sm')
sen = sp(u"I like to play football. I hated it in my childhood though") 
# Word, Dependency parser and related word
for word in sen.noun_chunks:
    print(word.text, word.root.dep_, word.root.head.text)
    
### Stemming
# Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word
# Different types of Stemming algorithms
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
# Needs Tokenized var
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(lambda x: [porter_stemmer.stem(y) for y in x])
# Raw non-Tokenized variable
Df['comment_text'].apply(lambda x: [porter_stemmer.stem(y) for y in x.split()])

# When variable in tokenized
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
Df["Text_Var"] = Df["POSTagged_Text_Var"].apply(lambda x: [porter_stemmer.stem(y[0]) for y in x])

from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(lambda x: [lancaster_stemmer.stem(y) for y in x])

from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer(“english”)
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(lambda x: [snowball_stemmer.stem(y) for y in x])

# Custom stemming using RegexpStemmer
from nltk.stem import RegexpStemmer
Regexp_stemmer = RegexpStemmer('ing')
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(lambda x: [Regexp_stemmer.stem(y) for y in x])

### Lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x])
#2nd way: Using spacy
import spacy
nlp = spacy.load('en', parse = False, tag=False, entity=False)
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

### Most frequent/common words (Top 50)
Freqs = pd.Series(' '.join(Df["Text_Var"]).split()).value_counts()[:50]

### Least Frequent/ Rare words (Bottom 50)
Freqs = pd.Series(' '.join(Df["Text_Var"]).split()).value_counts()[-50:]

### Remove list of words
Df["Text_Var"] = Df["Text_Var"].apply(lambda x: " ".join(x for x in x.split() if x not in Freqs))

# Synonyms of a word
from nltk.corpus import wordnet 
synonyms = []
for syn in wordnet.synsets('WORD'):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print(synonyms)

# Antonyms of a word
from nltk.corpus import wordnet
antonyms = []
for syn in wordnet.synsets("small"):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(antonyms)


#FlashText - 


####### Text Visualization #######
# The distribution of top unigrams BEFORE removing stop words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)    #Im case of Bi-Grams: vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus); 
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['Review Text'], 20)
for word, freq in common_words:
    print(word, freq)
df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df1.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in review before removing stop words')

# The distribution of top unigrams AFTER removing stop words
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)   # In case of BI-GRAMS: vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['Review Text'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df2.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in review after removing stop words')

# The distribution of top part-of-speech tags of review corpus
blob = TextBlob(str(df['Review Text']))
pos_df = pd.DataFrame(blob.tags, columns = ['word' , 'pos'])
pos_df = pos_df.pos.value_counts()[:20]
pos_df.iplot(
    kind='bar',
    xTitle='POS',
    yTitle='count', 
    title='Top 20 Part-of-speech tagging for review corpus')









