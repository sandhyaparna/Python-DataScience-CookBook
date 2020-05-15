##### Word2Vec
from gensim.models import Word2Vec
model_ted = Word2Vec(sentences=sentences_ted, size=100, window=5, min_count=5, workers=4, sg=0)
# sentences_ted is 2-d array with element being a word of the sentence
# sentences: the list of split sentences.
# size: the dimensionality of the embedding vector
# window: the number of context words you are looking at
# min_count: tells the model to ignore words with total count less than this number.
# workers: the number of threads being used
# sg: whether to use skip-gram or CBOW












