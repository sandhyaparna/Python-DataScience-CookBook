# https://ai.googleblog.com/2015/08/the-reusable-holdout-preserving.html
# https://github.com/bmcmenamin/thresholdOut-explorations/blob/master/Threshold%20out%20demos%20--%20tuning%20parameters%20for%20linear%20regression.ipynb

# Thresholdout
from numpy import *
def Thresholdout(sample, holdout, q):
   # function q is what you’re “testing” - e.g., model loss
   sample_mean = mean([q(x)  for x in sample])
   holdout_mean = mean([q(x)  for x in holdout])
   sigma = 1.0 / sqrt(len(sample))
   threshold = 3.0*sigma
   if (abs(sample_mean - holdout_mean)
          < random.normal(threshold, sigma) ):
       # q does not overfit: your “training estimate” is good
       return sample_mean
   else:
       # q overfits (you may have overfit using your training data)
       return holdout_mean + random.normal(0, sigma)
 
# 



       
       
       
