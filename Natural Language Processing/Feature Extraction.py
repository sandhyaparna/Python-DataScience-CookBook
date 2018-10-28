### CountVectorizer on Text var and not Tokenized var
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit(Df.Text_Var.values) # U can directly use X = vectorizer.fit_transform(Df.Text_Var.values)
Y = vectorizer.transform(Df.Text_Var.values)
vectorizer.get_feature_names()
Y.toarray()
# Word and its respective order in the 'vectorizer.get_feature_names()'
print(vectorizer.vocabulary_)
# Get Frequency list of all the words created
TextDf_CountVectors_Freq = pd.DataFrame({'Word':vectorizer.get_feature_names(), 'frequency':sum(Y).toarray()[0]}) 
# Count Vector as columns of a dataframe - Join it to the earlier data frame or use it a train data frame (cbind)
TextDf_CountVectors = pd.DataFrame(Y.A, columns=vectorizer.get_feature_names())

### N-Grams Vectorization
# ngram_range : tuple (min_n, max_n) - The lower and upper boundary of the range of n-values for different n-grams to be extracted. All # values of n such that min_n <= n <= max_n will be used
NGrams_vectorizer = CountVectorizer(ngram_range=(1,2))
X = NGrams_vectorizer.fit_transform(Df.Text_Var.values)
TextDf_NGramsVectors = pd.DataFrame(X.A, columns=NGrams_vectorizer.get_feature_names())
TextDf_NGramsVectors_Freq = pd.DataFrame({'Word':NGrams_vectorizer.get_feature_names(), 'frequency':sum(X).toarray()[0]})

### Character level Vectorization
Char_vectorizer = CountVectorizer(analyzer='char') #ngrams can also be added
X = Char_vectorizer.fit_transform(Df.Text_Var.values)
TextDf_CharVectors = pd.DataFrame(X.A, columns=Char_vectorizer.get_feature_names())
TextDf_CharVectors_Freq = pd.DataFrame({'Word':Char_vectorizer.get_feature_names(), 'frequency':sum(X).toarray()[0]})

### TF-IDF
1. Word level TF-IDF
TFIDF_vectorizer = TfidfVectorizer()
X = TFIDF_vectorizer.fit_transform(Df.Text_Var.values)
TextDf_TFIDFVectors = pd.DataFrame(X.A, columns=TFIDF_vectorizer.get_feature_names())
TextDf_TFIDFVectors_Freq = pd.DataFrame({'Word':TFIDF_vectorizer.get_feature_names(), 'frequency':sum(X).toarray()[0]}) 

2. ngram level TF-IDF
TFIDFNGrams_vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = TFIDFNGrams_vectorizer.fit_transform(Df.Text_Var.values)
TextDf_TFIDFNGramsVectors = pd.DataFrame(X.A, columns=TFIDFNGrams_vectorizer.get_feature_names())
TextDf_TFIDFNGramsVectors_Freq = pd.DataFrame({'Word':TFIDFNGrams_vectorizer.get_feature_names(), 'frequency':sum(X).toarray()[0]}) 

3. characters level TF-IDF - Single letter level 
TFIDFChar_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,2)) #ngrams can also be added
X = TFIDFChar_vectorizer.fit_transform(Df.Text_Var.values)
TextDf_TFIDFCharVectors = pd.DataFrame(X.A, columns=TFIDFChar_vectorizer.get_feature_names())
TextDf_TFIDFCharVectors_Freq = pd.DataFrame({'Word':TFIDFChar_vectorizer.get_feature_names(), 'frequency':sum(X).toarray()[0]}) 

### Co-Occurence Matrix
TextDf_CoOccurence = TextDf_CountVectors.astype(int) # TextDf_CountVectors is Data Frame from Count Vectorizer
TextDf_CoOccurence = TextDf_CoOccurence.T.dot(TextDf_CoOccurence)
np.fill_diagonal(TextDf_CoOccurence.values, 0) #Don't assign. Here automatically TextDf_CoOccurence DataFrame is modified. 












