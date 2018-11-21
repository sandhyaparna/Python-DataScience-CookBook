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


# Applying cross_val_score doesn't fit the DecisionTreeModel using the given Independent & Dependent data
# We need to fit the model
DecisionTreeModel = DecisionTreeModel.fit(X,y)


# Only after the DecisionTreeModel is fit using the data - attributes & Methods in the below link can be applied
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
 
DecisionTreeModel.feature_importances_ # Produces an imp value/factor for each variable in the input data
DecisionTreeModel.tree_ #<sklearn.tree._tree.Tree at 0x1af4e920> ???

DecisionTreeModel.apply(X)  #Gives the number of the node in the Decision tree where a particular OBSERVATION ended
DecisionTreeModel.decision_path(X) #Doesn't seem to work, checkout later ???
DecisionTreeModel.get_params() #Get parameters for fitted estimator
DecisionTreeModel.predict(Test_X) #Predict on the test data
DecisionTreeModel.predict_proba(Test_X) #Predict on the test data - Gives array with no of columns=no of labels in Target
DecisionTreeModel.predict_log_proba(Test_X) #Predict on the test data - Gives array with no of columns=no of labels in Target
DecisionTreeModel.score(Test_X,Test_y) # Mean Accuracy of Test Data

# Decision Tree Viz- Works when Target Var is both Cat/Num
# http://webgraphviz.com/
# Use the code from the saved dot file in the above link - it gives viz
data_feature_names = list(X)
target_names = pd.unique(y)
# Convert data type
target_names = target_names.astype('<U10') # Works when Target Var is both Cat/Num
# Viz code to a file - paste code in http://webgraphviz.com/
# DecisionTreeModel is one that is fitted on X,y
dotfile = open("path/DecisionTreeModel_Tree.dot", 'w')
tree.export_graphviz(DecisionTreeModel, out_file=dotfile, feature_names=data_feature_names,
                     class_names=target_names,filled=True, rounded=True)
dotfile.close()


    
    
    













