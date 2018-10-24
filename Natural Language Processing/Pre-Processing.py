import nltk
from nltk import *

### Noise Removal
# 1.Removing stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # stop_words = nltk.corpus.stopwords.words('english')
new_stop_words = ['stopWord1','stopWord2']
stop_words.extend(new_stop_words)
# Convert text data into small letter and split into words - words are seperated by comas
# Removes stop words from the Text var - Text_Var1 has words seperated by comas
 Df["Text_Var1"] = Df["Text_Var"].str.lower().str.split()
 Df["Text_Var1"] = Df["Text_Var1"].apply(lambda x: [item for item in x if item not in stop_words])
# Remove stop words from the text with adding comas between words
Df["Text_Var2"] = Df["Text_Var"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

