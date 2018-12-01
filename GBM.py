# Parameter Tuning
# https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/
# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-gradient-boosting-3363992e9bae

pd.options.display.max_rows = 1500
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199

gblinear booster treats missing values as zeros???

## Shrinkage parameter - Vary shrinkage parameter(all other parameters are default) to check for Test AUC, choose shrinkage parameter 
# that gives highest Test AUC and then vary number of trees to avoid over-fitting. Optimal values usually lie between 0.01-0.2
# Plot of AUC vs Learning Rates (2 curves - 1 for Training and another for Test)
# Every single time this below code is run, for same Train and Test datasets - the AUC values changes for a particular learning rate
learning_rates = [1, 0.5, 0.25, 0.1, 0.05, 0.01]
train_results = []
test_results = []
for eta in learning_rates:
   model = GradientBoostingClassifier(learning_rate=eta)
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
line1, = plt.plot(learning_rates, train_results, ‘b’, label='Train AUC')
line2, = plt.plot(learning_rates, test_results, ‘r’, label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('learning rate')
plt.show()








