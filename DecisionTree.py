# CANNOT handle MISSING values
# CANNOT handle categorical data - Should be encoded using different encoding techniques
# Target Variable can be BinaryLabel/MultiLabel and Numeric/Character

# scoring="roc_auc" works only when Target Var is BinaryNumeric
# But If no 'scoring' parameter is mentioned, it produces roc-auc by default for any type of Target var

# cross_val_predict function should have method='predict' & not method='predict_proba'
# method='predict_proba' - Generates array with 2 columns for Binary, 3 columns for Multi (columns correspond to the classes in sorted order)
# method='predict' - Generates array with column and different predicted categories(Original Categories as inputed for training)


DecisionTreeModel = DecisionTreeClassifier()
# single evaluation metric
DecisionTreeModel_scores = cross_val_score(DecisionTreeModel, X, y, cv=10, scoring="roc_auc")
DecisionTreeModel_scores # Gives evaluation metrics for eah cv set
print(score,":", "{:.3f} (std: {:.3f})".format(DecisionTreeModel_scores.mean(),DecisionTreeModel_scores.std()))

# Different evaluation metrics for each set of CV
# Multiple evaluation metrics on diff algos - https://stackoverflow.com/questions/35876508/evaluate-multiple-scores-on-sklearn-cross-val-score
scores=["roc_auc","accuracy", "precision", "recall","neg_log_loss","explained_variance"]
for score in scores:
    print (score,
    ":",
    "%.3f" % cross_val_score(DecisionTreeModel, X, y, cv=10, scoring=score).mean(),
    " ( std:",
    "%.3f" % cross_val_score(DecisionTreeModel, X, y, cv=10, scoring=score).std(),
    ")")

# method='predict' - Prediction on the test datesets within each set of Cross-validation
y_Pred = cross_val_predict(DecisionTreeModel,  X, y, cv=10, method='predict')
confusion_matrix(y, y_Pred) 

# method='predict_proba' - Prediction on the test datesets within each set of Cross-validation
# When a tree is too deep, a leaf is likely to contain only one single example - And hence probabilities will just be 0 for 0 and 1 for 1
y_Pred = cross_val_predict(DecisionTreeModel,  X, y, cv=10, method='predict_proba') #Produces array with number of columns=number of Labels
# Extract only 2nd column of the array i.e Prob of 1
y_Pred = Bank_y_train_Pred[:,1]
y_Pred = np.where(y_Pred>Confidence_value,1,0)




    
    
    













