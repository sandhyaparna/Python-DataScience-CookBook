import nltk
from nltk import *

### Noise Removal
# 1.Removing stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # stop_words = nltk.corpus.stopwords.words('english')
new_stop_words = ['stopWord1','stopWord2']
stop_words.extend(new_stop_words)
# Convert text data into small letter and split into words
Df["Text_Var"] = Df["Text_Var"].str.lower().str.split()
# Removes stop words from the Text var
Df["Text_Var"] =Df["Text_Var"].apply(lambda x: [item for item in x if item not in stop_words])

