# CAN handle MISSING values (DMatrix also takes missing values) 
# CAN handle categorical data - Should be encoded using different encoding techniques
# Target Variable can be BinaryLabel/MultiLabel and Numeric/Character

# https://www.analyticsvidhya.com/blog/2017/08/catboost-automated-categorical-data/
# https://tech.yandex.com/catboost/doc/dg/concepts/about-docpage/
# https://www.kdnuggets.com/2018/03/catboost-vs-light-gbm-vs-xgboost.html
# https://www.kdnuggets.com/2018/11/mastering-new-generation-gradient-boosting.html
# https://www.kdnuggets.com/2018/02/5-machine-learning-projects-overlook-feb-2018.html

# Parameters 
# https://tech.yandex.com/catboost/doc/dg/concepts/python-reference_parameters-list-docpage/
# https://tech.yandex.com/catboost/doc/dg/concepts/parameter-tuning-docpage/
# https://effectiveml.com/using-grid-search-to-optimise-catboost-parameters.html

import catboost
from catboost import *

CatBoostModel = catboost.CatBoostClassifier()
# cat_features parameter is a one-dimensional array of categorical columns indices.
CatBoostModel = CatBoostModel.fit(X,y,cat_features=list(range(6,15)))



