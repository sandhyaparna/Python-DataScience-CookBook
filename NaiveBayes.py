pd.options.display.max_rows = 1500
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

import sklearn as sklearn
from sklearn import *
from sklearn.naive_bayes import *

### Gaussian Naive Bayes ###
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
# Only Numeric Variables - and they should follow Normal distribution
Df_X = Df[['Var1','Var2','Var3']]

# Target Variable - Coding is not required for Target variable
Df_y = Df['Target_Var'] #Multi-class

# Alogorithm
Gaussian = GaussianNB()
Gaussian = Gaussian.fit(Df_X,Df_y)  # partial_fit is used to fit on batch of samples, if data is very huge
Df_y_Pred = Gaussian.predict(Df_X)
Gaussian.score(Bank_X,Bank_y) # Returns mean accuracy

### Multinomial Naive Bayes ###
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
# MultinomialNB states that: With a multinomial event model, samples (feature vectors) represent the frequencies with which certain events have been generated by a multinomial ... where p i is the probability that event i occurs. A feature vector ... is then a histogram, with x i {\displaystyle x_{i}} x_{i} counting the number of times event i was observed in a particular instance. This is the event model typically used for document classification, with events representing the occurrence of a word in a single document (see bag of words assumption)

# Features usually consists of occurence count i.e word counts for text processing, tf-idf may also work. 
# Categorical variable that is encoded usually is not the right approach. Categorical variables can be used in Bernouli by using Binary coding

# Var1, var2 etc have word counts or similar/Vars are not numeric but discret
Df_X = Df[['Var1','Var2','Var3']] 

# Target Variable - Coding is not required for Target variable
Df_y = Df['Target_Var'] #Multi-class

# Alogorithm
Multinomial = MultinomialNB()

### Bernoulli Naive Bayes ###
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

# Var1, var2 etc have binary/boolean values - 0 or 1
Df['Var1'] = Df['Var1'].map({'yes': 1, 'no': 0})
Df_X = Df[['Var1','Var2','Var3']] 

# Target Variable - Coding is not required for Target variable
Df_y = Df['Target_Var'] #Multi-class

# Alogorithm
Bernoulli = BernoulliNB()

### Complement Naive Bayes ###
# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB

# Designed to correct the “severe assumptions” made by the standard Multinomial Naive Bayes classifier - Used for imbalanced data sets 




