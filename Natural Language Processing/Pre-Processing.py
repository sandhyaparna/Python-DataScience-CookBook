import nltk
from nltk import *

### Noise Removal
# 1.Removing stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))  # stop_words = nltk.corpus.stopwords.words('english')
new_stop_words = ['stopWord1','stopWord2']
stop_words.extend(new_stop_words)

# Word Tokenization should be performed first and the removal of stop words - As Text may have sentences and punctuations are split in tokenization
# Convert text data into small letter
Df["Text_Var1"] = Df["Text_Var"].str.lower()
# Word Tokenization - words are seperated by commas
Df["Text_Var2"] = Df["Text_Var1"].apply(nltk.word_tokenize)   # Df["Text_Var2"] = Df["Text_Var"].str.lower().str.split() -- Same as Tokenization but . and , are attached to words as is in the text
# Removes stop words from the Text var2 i.e Tokenazied column
Df["Text_Var3"] = Df["Text_Var2"].apply(lambda x: [item for item in x if item not in stop_words])

# Remove stop words from the text with adding comas between words
Df["Text_Var4"] = Df["Text_Var1"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))






