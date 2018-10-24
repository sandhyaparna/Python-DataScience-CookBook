import nltk
from nltk import *

### Noise Removal
# Tokenization - splitting text into words (Removal of stop words, etc)
# http://www.nltk.org/api/nltk.tokenize.html - Different types of tokenization 

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))  # stop_words = nltk.corpus.stopwords.words('english')
new_stop_words = ['stopWord1','stopWord2']
stop_words.extend(new_stop_words)

# Word Tokenization should be performed first and the removal of stop words - As Text may have sentences and punctuations are split in tokenization

# Convert text data into small letter
Df["Text_Var1"] = Df["Text_Var"].str.lower()

# Extract Hashtagged words into a seperate column - Should not be Tokenized
Df["Text_Var5"] = Df["Text_Var"].str.findall(r'#.*?(?=\s|$)')

# Remove numbers from Text data - Should not be Tokenized
Df["Text_Var5"] = Df["Text_Var"].str.replace('\d+', '')

# Remove punctuations from strings
import string
s = '... some string with punctuation ...'
s = s.translate(None, string.punctuation)
Df["Text_Var"] = Df["Text_Var"].apply(lambda x: x.translate(None, string.punctuation))

# Remove Numbers from strings
Df["Text_Var"] = Df["Text_Var"].apply(lambda x: x.translate(None, string.digits))

# Word Tokenization - words are seperated by commas
Df["Text_Var2"] = Df["Text_Var1"].apply(nltk.word_tokenize)   # Df["Text_Var2"] = Df["Text_Var"].str.lower().str.split() -- Same as Tokenization but . and , are attached to words as is in the text

# Removes stop words from the Text var2 i.e Tokenazied column
Df["Text_Var3"] = Df["Text_Var2"].apply(lambda x: [item for item in x if item not in stop_words])

# Removes punctuations,# as well as tokenize - Within RegexpTokenizer function any expression string can be used 
tokenizer = RegexpTokenizer(r'\w+') #Alpha-numeic
Df["Text_Var4"] = tokenizer.tokenize(Df["Text_Var"])

# Tokenizing
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

# Remove stop words from the text with adding comas between words
Df["Text_Var4"] = Df["Text_Var1"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

### Stemming
# Stemming is a rudimentary rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word
# Different types of Stemming algorithms
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(lambda x: [porter_stemmer.stem(y) for y in x])
from nltk.stem.lancaster import LancasterStemmer
lancaster_stemmer = LancasterStemmer()
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(lambda x: [lancaster_stemmer.stem(y) for y in x])
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer(“english”)
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(lambda x: [snowball_stemmer.stem(y) for y in x])

### Lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()










