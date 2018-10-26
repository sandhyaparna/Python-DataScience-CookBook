# Python 3 Text Processing with NLTK Cookbook
# https://ucilnica.fri.uni-lj.si/pluginfile.php/46018/mod_resource/content/1/Python%203%20Text%20Processing%20with%20NLTK%203%20Cookbook.pdf
# https://www.analyticsvidhya.com/blog/2015/06/regular-expression-python/ - Regular Expressions
# https://www.analyticsvidhya.com/blog/2014/11/text-data-cleaning-steps-python/ - Data Cleaning

import nltk
from nltk import *

### Noise Removal
# Tokenization - splitting text into words (Removal of stop words, etc)
# http://www.nltk.org/api/nltk.tokenize.html - Different types of tokenization 

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

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

# Remove Hashtag words from text
Df["Text_Var"] = Df["Text_Var"].str.replace('#[\w]*', '')

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

# Spelling Correction
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
 if suggestions and edit_distance(word, suggestions[0]) <=
 self.max_dist:
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
replacer = CsvWordReplacer('synonyms.csv')

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
# Custom stemming using RegexpStemmer
from nltk.stem import RegexpStemmer
Regexp_stemmer = RegexpStemmer('ing')
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(lambda x: [Regexp_stemmer.stem(y) for y in x])

### Lemmatization
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x])


### POS (Part of speech) tagging 
# Single Sentence
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(nltk.pos_tag)
# Sentences
Df["Text_Var"] = Df["Sent_Tokenized_Text_Var"].apply(nltk.sent_tokenize)
# Build pos tagger using Treebank data
from nltk.corpus import treebank
from nltk.tag import tnt
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(treebank.tagged_sents())
# Save as pickle and then import it to use it (or) run above code and use the function
Df["Text_Var"] = Df["Tokenized_Text_Var"].apply(tnt_pos_tagger.tag)



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


