from sklearn.metrics import *

# Different metrrics for classification, Reg, clustering etc - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

# Stability of a model is very important i.e model should have similar performance metrics across both training and validation.
# Comparision of both training & test performance helps us to identify overfit models

# True Positives = Actual True & Predicted True
# False Positives = Actual False & Predicted True - False Alarm/Type I error
# False Negatives = Actual True & Predicted False - Missed Cases/Type II error
# True Negatives = Actual False & Predicted False - Correct Rejection

# Sensitivity/Recall/Hit rate/True Positive rate = TP/Overall Actual Positives = (1-False Positive rate)
# Specificity/Selectivity/True Negative rate = TN/Overall Actual Negatives
# Precision or PPV = TP/Predicted Positive
# Accuracy/Concordance = (TP+TN)/Total cases
# Misclassification rate = (FP+FN)/Total cases
# F1 Score = Harmonic mean of Precision and sensitivity = (2.TP)/(2.TP+FP+FN) = 2.(Precision).(Recall)/(Precision+Recall)
# F0.5 Score = Weighs precision(PPV) higher than recall
# F2 SCore = Weighs precision(PPV) lower than recall
  # F_beta = [(1 + beta^2).(precision.recall)] / [(beta^2 .precision) + (recall)]
#  
  
# ROC Curve - Summarizes performance of a classifier over all possible thresholds. (TP rate vs FP rate) or (Sensitivity vs 1-Specificity)
# AUC is area under ROC curve - Tells how much the model is capable of distinguishing between classes / Tells how good is the model for a given observation
# For multi-class, we can plot N AUC-ROC curves N classes using 1 vs All methodology. For eg if u have 3 classes named X,Y,Z 
# 1st ROC is X against Y&Z, 2nd is Y against (X&Z), 3rd is Z against (X&Y)
# AUC of 70% desirable

# ROC curves are used when there are roughly equal number of observations for each class
# Precision-Recall curves should be used there is a moderate to large class imbalance

# Preciison-Recall vs Thresholds: As Decision threshold inc, Sensitivity/Recall dec, Specificity & Precision inc

# https://www3.nd.edu/~busiforc/handouts/DataMining/Lift%20Charts.html
# ---Observations are ordered based on decreasing order of Predicted probability and then deciles/groups are created
# Each group is x% of overall population 
# In Decision tree -  observations within each node will have a certain prediction probability and each node is a bar in lift chart

# https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/
# https://towardsdatascience.com/end-to-end-python-framework-for-predictive-modeling-b8052bb96a78

# Gains/Lift Chart - Graph inc
# Top few deciles helps in capturing most of the True cases
# (X-axis - % of All cases) vs (Y-axis - % of True Cases ) - Line Graph is the cummulative % of true cases by each cumulative decile/group

# Actual vs Predicted - Graph dec
# (X-axis - % of All cases) vs (Y-axis - Count of True Cases) - Line graph with 2 lines-one for actual and one for predicted, lines are not cumulative true cases but they are count of true cases wrt to that decile population

# Lift Chart - Graph dec
# http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html
# The lift chart shows how much more likely we are to receive positive responses than if we contact a random sample of customers. 
# Lift value of each decile - Cumulative % of True cases till that decile / Cumulative % of population till that decile
# (X-axis - % of population) vs (Y-axis - Lift value) - https://www.saedsayad.com/model_evaluation_c.htm

# K-S Kolomogorov Smirnov Chart - http://www.saedsayad.com/model_evaluation_c.htm
# https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/
# K-S is a measure of the degree of separation between the positive and negative distributions
# Here the observations are ordered in increasing order of Predicted probability to create deciles
# For each decile - Calculate counts, cumulative counts, cumulative count% of 1s and 0s for each decile
# K-S value for each decile is the difference between Cumulative 1s % and cumulatiev 0s % - Max k-s value within all deciles is the final K-S stat

# Cost Function - https://towardsdatascience.com/model-performance-cost-functions-for-classification-models-a7b1b00ba60
# False_Positives = abs(fpr * Actual Negatives)
# True_Positives = abs(tpr * Actual Positives)
# Specificity = 1-fpr
# True_Negatives = abs(Specificity*Actual Negatives)
# False_Negatives = abs(Total - False_Positives - True_Positives - True_Negatives)
# Cost Function (Incurred) = True_Positives*(Cost Incurred for intervention - Cost Saved)  +  False_Positives*(Cost incurred for intervention)  +
#                            False_Negatives*(Cost if there is NO intervention)

# Multi-class classification Problem
# Quadratic Weighted Kappa - which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, the metric may go below 0. The quadratic weighted kappa is calculated between the scores which are expected/known and the predicted scores. 
# 5 step breakdown for Weighted Kappa Metric
  # First, create a multi class confusion matrix O between predicted and actual ratings.
  #   # Second, construct a weight matrix w which calculates the weight between the actual and predicted ratings.
  # Third, calculate value_counts() for each rating in preds and actuals.
  #   # Fourth, calculate E, which is the outer product of two value_count vectors
  # Fifth, normalise the E and O matrix
  # Caclulate, weighted kappa as per formula

# Multi-class log loss
  
  
  

y - Target Variable
y_Pred - Predicted Target variable as category and NOT continuous prob value
y_pred_proba - Predicted Target variable as continuous prob value i.e. y_pred_proba[:,1]
y_pred_proba = y_pred_proba[:,1] #For Binary classification

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
labels = ['label1', 'label2'] # label1 & label2 should be the labels in the y i.e [0,1] for binary classification
cm = confusion_matrix(y, y_pred, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Confusion Matrix - Simple
confusion_matrix(y, y_Pred) #Here y_Pred is a category and not continuous prob value

tn, fp, fn, tp = confusion_matrix(y, y_Pred).ravel()
tn, fp, fn, tp
pd.crosstab(y, y_Pred, rownames=['True'], colnames=['Predicted'], margins=True)

classification_report(y, y_Pred) # Produces (precision,recall,f1-score,support) for each labels seperately and all labels combined
accuracy_score(y, y_Pred)
precision_score(y, y_Pred) # Doesnt work if y_Pred is Categorical
roc_auc_score(y, y_Pred) # Doesnt work if y_Pred is Categorical (Works 'y_pred_proba' also but other metrics wont work)
recall_score(y, y_Pred) # Doesnt work if y_Pred is Categorical
f1_score((y, y_Pred) # Doesnt work if y_Pred is Categorical - F-score is computed with the harmonic mean of precision and recall

# roc-curve is (sensitivity vs 1-Specificity) or (TruePositiveRate vs FalsePositiveRate)      
# FalsePositiveRate = 1-Specificity
# ROC Curve plot - Visualization makes more sense when prediction are made using method='predict_proba'
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html   
import matplotlib.pyplot as plt
%matplotlib inline
fpr, tpr, thresholds = roc_curve(y,y_pred_proba)
# create plot
plt.plot(fpr, tpr, label='ROC curve'+', AUC='+str(round(roc_auc_score(y, y_pred_proba).mean(),3)))
plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc="lower right")

# ROC Curve and threshold 
# Choose the TP rate you are interested in - from that point draw a vertical line - identify where the vertical line meets the threshold line(Corresponds to threshold value) 
# http://abhay.harpale.net/blog/machine-learning/threshold-tuning-using-roc/
fpr, tpr, thresholds = roc_curve(y,y_pred_proba)
roc_auc = auc(fpr, tpr) # compute area under the curve
# plot
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
# create the axis of thresholds (scores)
ax2 = plt.gca().twinx()
ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
ax2.set_ylabel('Threshold',color='r')
ax2.set_ylim([thresholds[-1],thresholds[0]])
ax2.set_xlim([fpr[0],fpr[-1]]) 
         
         
# roc curve using scikitplot
import scikitplot
from scikitplot.metrics import *
plot_roc(y, y_pred_proba) #Here y_pred_proba is the original one with 2 columns of arrays for binary class and not y_pred_proba[:,1]
                  
# Precision Recall curve - Visualization makes more sense when prediction are made using method='predict_proba'
# predict_proba can be used in precision_recall_curve function but not in recall_score,precision_score functions
precision, recall, thresholds = precision_recall_curve(y,y_pred_proba)
# create plot 
plt.plot(precision, recall, label='Precision-recall curve'+', recall='+str(round(recall_score(y, y_Pred).mean(),3))+', precision='+str(round(precision_score(y, y_Pred).mean(),3)))
_ = plt.xlabel('Precision')
_ = plt.ylabel('Recall')
_ = plt.title('Precision-recall curve')

# plot_precision_recall vs threshold
precision, recall, thresholds = precision_recall_curve(y,y_pred_proba)
# Create plot
plt.figure(figsize=(8, 8))
plt.title("Precision and Recall Scores as a function of the decision threshold")
plt.plot(thresholds, precision[:-1], "b--", label="Precision")
plt.plot(thresholds, recall[:-1], "g-", label="Recall")
plt.ylabel("Score")
plt.xlabel("Decision Threshold")
plt.legend(loc='best')      
         
# Precision Recall curve using scikitplot
import scikitplot
from scikitplot.metrics import *
plot_precision_recall_curve(y, y_pred_proba) #Here y_pred_proba is the original one with 2 columns of arrays for binary class and not y_pred_proba[:,1]

# Confusion Matrix using scikitplot         
plot_confusion_matrix(y, y_Pred, normalize=True)

# roc-curve using scikitplot
plot_roc_curve(y, y_pred_proba)
plot_roc(y, y_pred_proba)

# Precision Recall curve  using scikitplot
plot_precision_recall_curve(y, y_pred_proba)
plot_precision_recall(y, y_pred_proba)

# Gains/Lift Chart  using scikitplot        
plot_cumulative_gain(y, y_pred_proba)

# Lift Chart using scikitplot         
plot_lift_curve(y, y_pred_proba)         

# K-S Kolomogorov Smirnov Chart using scikitplot
plot_ks_statistic(y, y_pred_proba)
       
         
         
# For clustering
def plot_silhouette(X, cluster_labels, title='Silhouette Analysis',
                    metric='euclidean', copy=True, ax=None, figsize=None,
                    cmap='nipy_spectral', title_fontsize="large",
                    text_fontsize="medium")
         

# Concordant-Discordant ratio
        





