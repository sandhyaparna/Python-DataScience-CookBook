# CAN handle MISSING values
# CANNOT handle categorical data - Should be encoded using different encoding techniques
# Target Variable can be BinaryLabel/MultiLabel and Numeric/Character

# Parameter Tuning
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# https://xgboost.readthedocs.io/en/latest/index.html
# https://xgboost.readthedocs.io/en/latest/python/python_intro.html
# https://xgboost.readthedocs.io/en/latest/python/python_api.html
# https://xgboost.readthedocs.io/en/latest/parameter.html
# Explains how to handle imbalnced datsets in xgboost - https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html


## xgb.train/xgb.cv is used probably for faster processing
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














  
  
  
