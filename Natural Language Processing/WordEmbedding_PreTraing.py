##### Word2Vec
# Implementation of Word2Vec and FastText https://towardsdatascience.com/word-embedding-with-word2vec-and-fasttext-a209c1d3e12c
from gensim.models import Word2Vec
model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=4, sg=0)
# sentences_ted is 2-d array with element being a word of the sentence
# sentences: the list of split sentences.
# size: the dimensionality of the embedding vector
# window: the number of context words you are looking at
# min_count: tells the model to ignore words with total count less than this number.
# workers: the number of threads being used
# sg: whether to use skip-gram or CBOW
model_ted.wv.most_similar(“man”) #Vector for Man





##### Pre-trained word2vec model is used to find distance between 2 sentences
# https://towardsdatascience.com/finding-similar-quora-questions-with-word2vec-and-xgboost-1a19ad272c0d
question1 = 'What would a Trump presidency mean for current international master’s students on an F1 visa?'
question2 = 'How will a Trump presidency affect the students presently in US or planning to study in US?'
question1 = question1.lower().split()
question2 = question2.lower().split()
question1 = [w for w in question1 if w not in stop_words]
question2 = [w for w in question2 if w not in stop_words]
from gensim.models import Word2Vec
model = gensim.models.KeyedVectors.load_word2vec_format('./word2Vec_models/GoogleNews-vectors-negative300.bin.gz', binary=True)
model.init_sims(replace=True) #Normalized Word Mover's Distance (WMD)
distance = model.wmdistance(question1, question2)
print('distance = %.4f' % distance) # If Distance is more then sentences are not similar to each other

def norm_wmd(q1, q2):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    stop_words = stopwords.words('english')
    q1 = [w for w in q1 if w not in stop_words]
    q2 = [w for w in q2 if w not in stop_words]
    return norm_model.wmdistance(q1, q2)
df['norm_wmd'] = df.apply(lambda x: norm_wmd(x['question1'], x['question2']), axis=1) # On a datset where question1, question2 are variables

##### FastText
from gensim.models import FastText
model_ted = FastText(sentences_ted, size=100, window=5, min_count=5, workers=4,sg=1)
model_ted.wv.most_similar("Gastroenteritis")


##### ULMFit - Transfer Learning technique
# https://colab.research.google.com/drive/1NMaMt94_shDH7kTktC5yvgAKsYfQBEHF

##### ELMo 
# ELMo from Scratch - Training on Custom data - https://appliedmachinelearning.blog/2019/11/30/training-elmo-from-scratch-on-custom-data-set-for-generating-embeddings-tensorflow/
# Using pre-trained ELMo - https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/

##### Word2Vec 
# Word2vec from Scratch - Training on Custom data - https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
# https://www.kaggle.com/chewzy/tutorial-how-to-train-your-custom-word-embedding

##### Transformer 
# Implementation in Colab https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb#scrollTo=s19ucTii_wYb





















