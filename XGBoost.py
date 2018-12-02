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

# xgboost.XGBClassifier - https://xgboost.readthedocs.io/en/latest/python/python_api.html















  
  
  
