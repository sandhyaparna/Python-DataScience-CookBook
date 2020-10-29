### Bag of words - CountVectorizer on Text var and not Tokenized var
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
## simple code
X = vectorizer.fit_transform(TrainData.comment_text.values)
pd.DataFrame(X.A, columns=vectorizer.get_feature_names())
#
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
# ngram_range : tuple (min_n, max_n) - The lower and upper boundary of the range of n-values for different n-grams to be extracted. 
# All # values of n such that min_n <= n <= max_n will be used
NGrams_vectorizer = CountVectorizer(ngram_range=(1,2))
X = NGrams_vectorizer.fit_transform(Df.Text_Var.values)
TextDf_NGramsVectors = pd.DataFrame(X.A, columns=NGrams_vectorizer.get_feature_names())
TextDf_NGramsVectors_Freq = pd.DataFrame({'Word':NGrams_vectorizer.get_feature_names(), 'frequency':sum(X).toarray()[0]})

### Character level Vectorization - single letter based
Char_vectorizer = CountVectorizer(analyzer='char') #ngrams can also be added
X = Char_vectorizer.fit_transform(Df.Text_Var.values)
TextDf_CharVectors = pd.DataFrame(X.A, columns=Char_vectorizer.get_feature_names())
TextDf_CharVectors_Freq = pd.DataFrame({'Word':Char_vectorizer.get_feature_names(), 'frequency':sum(X).toarray()[0]})

### TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

1. Word level TF-IDF
TFIDF_vectorizer = TfidfVectorizer()
X = TFIDF_vectorizer.fit_transform(Df.Text_Var.values) #fit_transform on Train data
Xtest = TFIDF_vectorizer.transform(Test) #trnaform on Test data
TextDf_TFIDFVectors = pd.DataFrame(X.A, columns=TFIDF_vectorizer.get_feature_names())
TextDf_TFIDFVectors = pd.DataFrame(np.round(X.A,2), columns=TFIDF_vectorizer.get_feature_names()) #To get 2 decimal places only
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
Df["Text_Var_SentimentValue_Polarity"] = Df["Text_Var"].apply(lambda x: textblob.TextBlob(x).sentiment[0])
Df["Text_Var_SentimentValue_Polarity"] = Df["Text_Var"].apply(lambda x: textblob.TextBlob(x).sentiment[1])
# Below is a slow processing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
Df["polarity"].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x))
Df["Negativity"].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)["neg"])
Df["Neutrality"].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)["neu"])
Df["Positivity"].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)["pos"])
Df["OverallSentiment"].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)["compound"])

### Readability scores
# The Flesch readability ease is an indicator designed to quantify how difficult a passage is to understand. The Flesch readability ease is calculated mathematically using the formula below:
# From the above formula, we can see that comments with shorter words and fewer words per sentence are more "readable".
!pip install textstat
TrainData["flesch_reading_ease"] = TrainData["comment_text"].progress_apply(textstat.flesch_reading_ease)
TrainData["automated_readability"] = TrainData["comment_text"].progress_apply(textstat.automated_readability_index)
TrainData["dale_chall_readability"] = TrainData["comment_text"].progress_apply(textstat.dale_chall_readability_score)


### Topic Modeling (Latent Dirichlet Allocation) as features
Data link - https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
https://www.analyticsvidhya.com/blog/2018/10/stepwise-guide-topic-modeling-latent-semantic-analysis/
https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
Data from the above website - https://gist.github.com/kunalj101/ad1d9c58d338e20d09ff26bcc06c4235
# First apply count vectorization (or) Tf-idf
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(Df.Text_Var.values)
#
from sklearn import decomposition
from sklearn.decomposition import *
# Train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=25, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(X)
# Add variable names to features, to determine which topic/feature has more value
# Create a list of Variable names in list like Var1,Var2,Var3,Var4 etc
Topic_names = []
for i in range(1,n+1):
    Topic_names += ["Topic"+str(i)]
# Dataframe with number of Topics extracted as number of variables. Topic1 var gives each documents weightage for Topic1, etc.    
TextDf_LDA = pd.DataFrame(X_topics, columns=Topic_names)\
# view the topic models - top words in each topic as a summary
topic_word = lda_model.components_ 
vocab = vectorizer1.get_feature_names()
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
topic_summaries

### Document Clustering with Similarity Features
https://towardsdatascience.com/understanding-feature-engineering-part-3-traditional-methods-for-text-data-f6f7d70acd41
https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
# First apply count vectorization (or) Tf-idf
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(Df.Text_Var.values)
# Similarity Matrix
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(X.A)
# Clustering on Similarity Matrix
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(similarity_matrix, 'ward') # Takes app. 5-10mins for processing 10,000 records of text
# Z is matrix with 4 columns, first 2 columns are the final clusters(only 2 in hierarchical clustering), 3rd column is the distance 
# between final 2 clusters - From Z we can create our own clusters using distance parameter
plt.figure(figsize=(8, 3))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point') 
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y=50, c='k', ls='--', lw=0.5) # y value is changed based on the distance at which you want clusters to be split
# Create clusters at the specified distance
rom scipy.cluster.hierarchy import fcluster
max_dist = 50
# Extract cluster labels from Linkage matrix based on the max_dist
cluster_labels = fcluster(Z, max_dist, criterion='distance')
# Convert above array into Dataframe and assign column name as ClusterLabel
cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
# Attact cluster_labels dataframe to out original DataFrame with Text column
Df_Clustered = pd.concat([Df, cluster_labels], axis=1)

### Named Entity Recognition
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []

    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    if current_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)
            current_chunk = []
    return continuous_chunk
Df["TextVar_Chunked"] = Df["Text_Var"].apply(get_continuous_chunks)

## Using spacy
import spacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
# Gives tokens for all the words in the column - corpus_df["Document"]
tokens = nlp(''.join(str(Df["Text_Var"].tolist())))
# For tokens generated above, if the words are part of entities category, those words are outputed using below code
for ent in tokens.ents:
    print(ent.text, ent.label_)
# Visualize named entities with displacy
spacy.displacy.render(tokens, style='ent', jupyter=True)
# Explain Category
spacy.explain('WORK_OF_ART')
spacy.explain('NORP')
# Most freq word tokens
items = [x.text for x in tokens.ents]
Counter(items).most_common(20)
# Extract only Person category
person_list = []
for ent in tokens.ents:
    if ent.label_ == 'PERSON':
        person_list.append(ent.text)
person_counts = Counter(person_list).most_common(20)
df_person = pd.DataFrame(person_counts, columns =['text', 'count'])
# Extract only NORP (nationalities, religious and political groups) category
norp_list = []
for ent in tokens.ents:
    if ent.label_ == 'NORP':
        norp_list.append(ent.text)
norp_counts = Counter(norp_list).most_common(20)
df_norp = pd.DataFrame(norp_counts, columns =['text', 'count'])

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


### Step by step implementation of CBOW, Skip-gram & word2vec
https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa

### Continuous Bag-of-Words (CBOW)

### Training Word2vec model from documents
https://medium.freecodecamp.org/how-to-get-started-with-word2vec-and-then-how-to-make-it-work-d0a2fca9dad3

### Pre-trained Wikipedia corpus Word2vec
Training of Word2vec on Wiki Corpus   
First, we need process the xml format wikipedia to text format
https://textminingonline.com/training-word2vec-model-on-english-wikipedia-by-gensim
https://medium.com/@black_swan/how-to-train-word2vec-and-fasttext-embedding-on-wikipedia-corpus-9e8ac45a0c0a    
Train and Update Word2vec 
https://medium.com/@maxminicherrycc/word2vec-how-to-train-and-update-it-4eed4260cf75 <br/>
    
### Google News Word2vec model
# https://github.com/sandhyaparna/Python-DataScience-CookBook/blob/master/Natural%20Language%20Processing/NLP_notebook.ipynb
# Save Google's pre trained model from - https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# STEPS
# Open Anaconda Prompt - Run as Administrator
# In Anconda Prompt - activate tensorflow_cpu
# Install necessary packages
# Open jupyter notebook 
import gensim
word2vec = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/sandh/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)

# Apply word2vec model on a sentence -  For each word in the sentence 300 values are generated
# Performance of word2vec is evaluated using similarity (2 words are inputted) & words nearest (word is mentioned, outputs the nearest words to the input word)
sentence = ["London", "is", "the", "capital", "Great", "Britain"]
vectors = [word2vec[w] for w in sentence]

# Apply averaging on the words of a sentence - For each sentence 300 values/vectors are generated
# Create empty data frame and add text data as in clean_questions -  https://github.com/hundredblocks/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb
columns = ['Unnamed','text','tokens']
df_ = pd.DataFrame( columns=columns)
my_dic = {'Unnamed':3, 'text':'just happened a terrible car crash', 'tokens':'[just, happened, a, terrible, car, crash]'}
df_.loc[len(df_)] = my_dic
my_dic = {'Unnamed':2, 'text':'our deeds are the reason of this earthquake', 'tokens':'[our, deeds, are, the, reason, of, this, earthquake]'}
df_.loc[len(df_)] = my_dic
my_dic = {'Unnamed':4, 'text':'there is a forest fire at spot pond, geese ', 'tokens':'[[there, is, a, forest, fire, at, spot, pond, geese]'}
df_.loc[len(df_)] = my_dic

# Coordinate-wise averaging
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)

# embeddings return 300 values/vectors for each sentence
embeddings = get_word2vec_embeddings(word2vec, df_)

# Word2Vec Gensim
# Download model from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
from gensim.models import Word2Vec
#loading the downloaded model
model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)
#the model is loaded. It can be used to perform all of the tasks mentioned above.
# getting word vectors of a word
dog = model['dog']
#performing king queen magic
print(model.most_similar(positive=['woman', 'king'], negative=['man']))
#picking odd one out
print(model.doesnt_match("breakfast cereal dinner lunch".split()))
#printing similarity index
print(model.similarity('woman', 'man'))


### Doc2vec implementation 
# Step by step implementation
https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5

### Glove Model







