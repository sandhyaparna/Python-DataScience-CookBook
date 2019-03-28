# CAN handle MISSING values (DMatrix also takes missing values) - XGBoost will automatically learn what is the best direction to go when a value is missing (or) 
  # Automatically "learn" what is the best imputation value for missing values based on reduction on training loss.
  # Incidentally, xgboost and lightGBM both treat missing values in the same way as xgboost treats the zero values in sparse matrices; 
  # it ignores them during split finding, then allocates them to whichever side reduces the loss the most.
# CANNOT handle categorical data - Should be encoded using different encoding techniques
# Target Variable can be BinaryLabel/MultiLabel and Numeric/Character

# Parameter Tuning
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# https://xgboost.readthedocs.io/en/latest/index.html
# https://xgboost.readthedocs.io/en/latest/python/python_intro.html
# https://xgboost.readthedocs.io/en/latest/python/python_api.html
# https://xgboost.readthedocs.io/en/latest/parameter.html
# http://mlexplained.com/2018/01/05/lightgbm-and-xgboost-explained/
# Explains how to handle imbalnced datsets in xgboost - https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html


## xgb.train/xgb.cv is used probably for faster processing
# xgboost implementation, prediction using xgb.train - https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
# Check this link to create DMatrix, apply xgb.cv, xgb.train etc - https://xgboost.readthedocs.io/en/latest/python/python_intro.html 
# Check xgboost.cv & xgboost.train in this link for parameter tuning - https://xgboost.readthedocs.io/en/latest/python/python_api.html

# xgb.cv - https://rdrr.io/cran/xgboost/man/xgb.cv.html
# xgb.cv - https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py

import xgboost as xgb
from xgb import *

# DMatrix is a internal data structure that used by XGBoost which is optimized for both memory efficiency and training speed
# DMatrix is used in xgb.cv, xgb.train, etc
# xgb.XGBClassifier is an sklearn version of xgboost and don't use DMatrix, uses regular data matrices
# n_estimators in xgb.XGBClassifier can be set to 100 if the datset is huge, 1000 if the dataset is medium

## Hyper parameter tuning using randomized grid search - RandomizedSearchCV
# GridSearchCV (exhaustive search) would take more time to get done
# https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
# https://scikit-learn.org/stable/modules/grid_search.html
# https://www.arunprakash.org/2017/04/gridsearchcv-vs-randomizedsearchcv-for.html

# xgboost.XGBClassifier - https://xgboost.readthedocs.io/en/latest/python/python_api.html
# In fit of xgboost.XGBClassifier, if early_stopping_rounds is mentioned , the fit of xgboost.XGBClassifier returns the model 
# from the last iteration (not the best one). If early stopping occurs, the model will have three additional fields: 
# bst.best_score, bst.best_iteration and bst.best_ntree_limit. (Use bst.best_ntree_limit to get the correct value if num_parallel_tree and/or num_class appears in the parameters)

# GridSearchCV
from sklearn.grid_search import GridSearchCV
# https://www.arunprakash.org/2017/04/gridsearchcv-vs-randomizedsearchcv-for.html
DecisionTreeModel_GridSearch = DecisionTreeClassifier(random_state=0)
parameter_grid = {"criterion": ["gini", "entropy"],
                  "splitter": ["best", "random"],
                  "max_depth": np.arange(4,6),
                  "min_samples_split": [40,60,70],
                  "min_samples_leaf": [15,20,25,30], #range(15,30,5)
                  "class_weight":["balanced"]}
gridSearch = GridSearchCV(DecisionTreeModel_GridSearch, param_grid=parameter_grid, cv=10, scoring="roc_auc")
gridSearch = gridSearch.fit(X, y) #fits the GridSearchCV to data given
gridSearch.grid_scores_ #Gives scores of each of the parameter_grid parameters
gridSearch.best_params_ #Parameters that gives the best performance
gridSearch.best_score_ #Test set results decide the best params
gridSearch.best_estimator_ #Parameters that gives the best performance fit to mentioned classifier
# U dont need to fit this model on the train data set - It is already fit in gridsearch.fit
DecisionTreeModel = gridSearch.best_estimator_ 
# Export/Save Model
pickle.dump(DecisionTreeModel,open("C:/Users/User/Google Drive (sandhya.pashikanti@uconn.edu)/Data Science/Python Learning/DecisionTreeModel.pkl","wb"))
# Import/Load Model
DecisionTreeModel = pickle.load(open("C:/Users/User/Google Drive (sandhya.pashikanti@uconn.edu)/Data Science/Python Learning/DecisionTreeModel.pkl","r"))

# RandomizedSearchCV
# https://www.arunprakash.org/2017/04/gridsearchcv-vs-randomizedsearchcv-for.html
# n_iter parameter needs to be mentioned in addition to the ones in GridSearchCV - out of the totals combinations only 10 randomly selected paras are run
from sklearn.grid_search import RandomizedSearchCV
# Code is similar to GridSearchCV

# XGBoost Important features - xgboost.plot_importance in https://xgboost.readthedocs.io/en/latest/python/python_api.html
# https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
# https://github.com/slundberg/shap
xgb.plot_importance(XGBoostModel) # weight is default importance_type
xgb.plot_importance(XGBoostModel,max_num_features=15 ) # Displays top 15 features
xgb.plot_importance(XGBoostModel,importance_type="weight") #weight is number of times a feature appears in a tree
xgb.plot_importance(XGBoostModel,importance_type="cover") #cover is the average coverage of splits which use the feature where coverage is defined as the number of samples affected by the split
xgb.plot_importance(XGBoostModel,importance_type="gain") #gain is the average gain of splits which use the feature

## Different methods of Feature importance
# http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine.html
# https://christophm.github.io/interpretable-ml-book/shapley.html#
## SHAP - Feature Importance
# Comparision of interpreting features using diff methods - http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine.html#SHAP
# SHAP on Finance data - https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
# SHAP on Health data - https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html
XGBoostModel = XGBClassifier()
XGBoostModel = XGBoostModel.fit(X,y)
# SHAP(SHapley Additive exPlanations). SHAP assigns each feature an importance value for a particular prediction. 
# Help measure the impact of the features on the predictions
# Tree SHAP method is mathematically equivalent to averaging differences in predictions over all possible orderings of the features, rather than just the ordering specified by their position in the tree.
import shap
# Below code produces shap value for each column of all observations 
shap_values = shap.TreeExplainer(XGBoostModel).shap_values(X)
# Bar chart of feature importance
shap.summary_plot(shap_values, X, plot_type="bar")
# Chart of feature importance for each observation
shap.summary_plot(shap_values, X) 
# Shap values of a particular feature vs Actual feature values - SHAP dependence plot show how the model output varies by feauture value
# The feature(second feature) used for coloring is automatically chosen to highlight what might be driving these interactions.
shap.dependence_plot("Var1",shap_values, X)
# SHAP dependence plot of all var names in X
for var in X.columns:
    shap.dependence_plot(var, shap_values, X)   
# SHAP Interaction Value Summary Plot
shap_interaction_values = shap.TreeExplainer(XGBoostModel).shap_interaction_values(X)
shap.summary_plot(shap_interaction_values, X)
# To choose 2nd feature by your choice and no automatic
shap.dependence_plot(("Var1", "Var1"),shap_interaction_values, X) #Var1 only
shap.dependence_plot(("Var1", "Var2"),shap_interaction_values, X) #choose Var2 instead of automatic choosing


  
  
  
