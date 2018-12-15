# CANNOT handle MISSING values
# CANNOT handle categorical data - Should be encoded using different encoding techniques
# Target Variable can be BinaryLabel/MultiLabel and Numeric/Character

# Used to check the fastest loop run time of a function
%timeit

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# Parameter Tuning
# https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

# max_features: Number of features allowed to try in individual tree.
# Increasing max_features generally improves the performance of the model but may overfit
# Decreasing max_features decrease the speed of algorithm
# Hence, you need to strike the right balance and choose the optimal max_features.

# n_estimators: Number of trees in the forest.
# Higher number of trees give you better performance but makes your code slower.
# You should choose as high value as your processor can handle because this makes your predictions stronger and more stable.

# min_sample_leaf: 
# A smaller leaf makes the model more prone to capturing noise in train data. 
# Generally,prefer a minimum leaf size of more than 50 but depends on the problem as well

# n_jobs: -1 means no restriction on number of processors it is allowed to use

# random_state: To replicate results

# oob_score: This method simply tags every observation used in different tress. And then it finds out a maximum vote score for every 
# observation based on only trees which did not use this particular observation to train itself.



