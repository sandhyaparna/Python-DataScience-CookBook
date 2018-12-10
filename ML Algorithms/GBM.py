# CANNOT handle MISSING values
# CANNOT handle categorical data - Should be encoded using different encoding techniques
# Target Variable can be BinaryLabel/MultiLabel and Numeric/Character

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

# Parameter Tuning
# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae

## learning_rate/Shrinkage parameter - Vary shrinkage parameter(all other parameters are default) to check for Test AUC, choose shrinkage parameter 
# that gives highest Test AUC and then vary number of trees to avoid over-fitting. Optimal values usually lie between 0.01-0.2

## After tuning learning_rate and no of estimators, vary tree-specific parameters

## sub-samples - The fraction of samples to be used for fitting the individual base learners. Selection is done by random sampling. 
# If smaller than 1.0 this results in Stochastic Gradient Boosting.
# At each iteration of the algorithm, a base learner should be fit on a subsample of the training set drawn at random without replacement, help prevent overfitting, acting as a kind of regularization.
# Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
# Typical values ~0.8 generally work fine but can be fine-tuned further. Values between 0.5-0.8 leads to good results for small and moderate sized training sets.

## loss -  ‘exponential’ performs AdaBoost algorithm

## random_state is used to make outcomes consistent across different runs using the same parameters

gblinear booster treats missing values as zeros???

GBMModel = GradientBoostingClassifier()
GBMModel = GBMModel.fit(X,y)

# Plot of AUC vs Learning Rates (2 curves - 1 for Training and another for Test)
# Every single time this below code is run, for same Train and Test datasets - the AUC values changes for a particular learning rate
# Setting random_state to a fixed value avoids this problem 
learning_rates = [1, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.025, 0.01]
train_results = []
test_results = []
for eta in learning_rates:
   model = GradientBoostingClassifier(learning_rate=eta,random_state=0)
   model.fit(x_train, y_train)
   train_pred = model.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc) # Training 
   y_pred = model.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc) # Test
  
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(learning_rates, train_results, label='Train AUC')
line2, = plt.plot(learning_rates, test_results, label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('learning rate')
plt.show()








