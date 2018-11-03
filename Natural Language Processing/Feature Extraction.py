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

### Sentiment of the Texts
import textblob
from textblob import TextBlob
Df["Text_Var_SentimentValue"] = Df["Text_Var"].apply(lambda x: TextBlob(x).sentiment[0])

### Topic Modeling (Latent Dirichlet Allocation) as features
Data link - https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
Data from the above website - https://gist.github.com/kunalj101/ad1d9c58d338e20d09ff26bcc06c4235
    
from sklearn import decomposition
from sklearn.decomposition import *
# Train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=25, learning_method='online', max_iter=20)

add variable names to features, to determine which topic/feature has more value

### Document Similarity
# First apply count vectorization (or) Tf-idf
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(Df.Text_Var.values)
X = X.toarray()
# 
from sklearn.metrics.pairwise import cosine_similarity




### Extract different part of speech word sets from Text_Var and append them to create a single var
# Import textblob.download_corpora
import textblob
from textblob import TextBlob
subprocess.check_call(["python", '-m', 'textblob.download_corpora'])
pos_family = {'noun' : ['NN','NNS','NNP','NNPS'], 'pron' : ['PRP','PRP$','WP','WP$'], 'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'], 'adv' : ['RB','RBR','RBS','WRB'] }
# function to check and get the part of speech tag count of a words in a given sentence
def pos_family_count(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
               cnt += 1
    except:
        pass
    return cnt
# Function to extract pos_family words
def pos_family_words(x, flag):
    pos_words = list()
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                w = list(tup)[0]   
                pos_words.append(w)
    except:
        pass
    return pos_words
#
Df['noun_count'] = Df['Text_Var'].apply(lambda x: pos_family_count(x, 'noun'))
Df['nouns'] = Df['Text_Var'].apply(lambda x: pos_family_words(x, 'noun'))

### Total number of letters/chars
Df['char_count'] = Df['Text_Var'].apply(len)

### Total number of words
Df['word_count'] = Df['Text_Var'].apply(lambda x: len(x.split()))

### Average length of the words used
Df['word_density'] = Df['char_count'] / (Df['word_count']+1)

### Number of Stop Words
from nltk.corpus import stopwords
stop = stopwords.words('english')
Df['stopwords'] = Df['Text_Var'].apply(lambda x: len([x for x in x.split() if x in stop]))

### Total number of punctuation marks
Df['punctuation_count'] = Df['Text_Var'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 

### Total number of upper count words
Df['title_word_count'] = Df['Text_Var'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))

### Total number of proper case (title) words 
Df['upper_case_word_count'] = Df['Text_Var'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

### Number of special characters
Df['hastags'] = Df['Text_Var'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

### Number of numerics
Df['numerics'] = Df['Text_Var'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))




