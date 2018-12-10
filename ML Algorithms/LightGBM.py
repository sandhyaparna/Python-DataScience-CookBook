# CAN handle MISSING values (DMatrix also takes missing values) - 
  # Incidentally, xgboost and lightGBM both treat missing values in the same way as xgboost treats the zero values in sparse matrices; 
  # it ignores them during split finding, then allocates them to whichever side reduces the loss the most.
# CANNOT handle categorical data - Should be encoded using different encoding techniques
# Target Variable can be BinaryLabel/MultiLabel and Numeric/Character

# Parameter Tuning
# https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
# https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
# http://mlexplained.com/2018/01/05/lightgbm-and-xgboost-explained/
# Similar to XGBoost - https://lightgbm.readthedocs.io/en/latest/

# https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
# Implemetation of lightgbm 

# Tuning Parameters of Light GBM
# For best fit
# num_leaves : This parameter is used to set the number of leaves to be formed in a tree. Theoretically relation between num_leaves and max_depth is num_leaves= 2^(max_depth). However, this is not a good estimate in case of Light GBM since splitting takes place leaf wise rather than depth wise. Hence num_leaves set must be smaller than 2^(max_depth) otherwise it may lead to overfitting. Light GBM does not have a direct relation between num_leaves and max_depth and hence the two must not be linked with each other.
# min_data_in_leaf : It is also one of the important parameters in dealing with overfitting. Setting its value smaller may cause overfitting and hence must be set accordingly. Its value should be hundreds to thousands of large datasets.
# max_depth: It specifies the maximum depth or level up to which tree can grow.
 
# For faster speed
# bagging_fraction : Is used to perform bagging for faster results
# feature_fraction : Set fraction of the features to be used at each iteration
# max_bin : Smaller value of max_bin can save much time as it buckets the feature values in discrete bins which is computationally inexpensive.
 
# For better accuracy
# # Use bigger training data
# num_leaves : Setting it to high value produces deeper trees with increased accuracy but lead to overfitting. Hence its higher value is not preferred.
# max_bin : Setting it to high values has similar effect as caused by increasing value of num_leaves and also slower our training procedure.
