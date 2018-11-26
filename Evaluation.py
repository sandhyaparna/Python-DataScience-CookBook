from sklearn.metrics import *

# Different metrrics for classification, Reg, clustering etc - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

# True Positives = Actual True & Predicted True
# False Positives = Actual False & Predicted True - False Alarm/Type I error
# False Negatives = Actual True & Predicted False - Missed Cases/Type II error
# True Negatives = Actual False & Predicted False - Correct Rejection

# Sensitivity/Recall/Hit rate/True Positive rate = TP/Overall Actual Positives = (1-False Positive rate)
# Specificity/Selectivity/True Negative rate = TN/Overall Actual Negatives
# Precision or PPV = TP/Predicted Positive
# Accuracy = (TP+TN)/Total cases
# Misclassification rate = (FP+FN)/Total cases
# F1 Score = Harmonic mean of Precision and sensitivity = (2.TP)/(2.TP+FP+FN) = 2.(Precision).(Recall)/(Precision+Recall)
# F0.5 Score = Weighs precision(PPV) higher than recall
# F2 SCore = Weighs precision(PPV) lower than recall
  # F_beta = [(1 + beta^2).(precision.recall)] / [(beta^2 .precision) + (recall)]

# ROC Curve - Summarizes performance of a classifier over all possible thresholds. (TP rate vs FP rate) or (Sensitivity vs 1-Specificity)
# AUC is area under ROC curve - Tells how much the model is capable of distinguishing between classes / Tells how good is the model for a given observation
# For multi-class, we can plot N AUC-ROC curves N classes using 1 vs All methodology. For eg if u have 3 classes named X,Y,Z 
# 1st ROC is X against Y&Z, 2nd is Y against (X&Z), 3rd is Z against (X&Y)

# https://www3.nd.edu/~busiforc/handouts/DataMining/Lift%20Charts.html
# ---Observations are ordered based on decreasing order of Predicted probability and then deciles/groups are created
# Each group is x% of overall population 
# In Decision tree -  observations within each node will have a certain prediction probability and each node is a bar in lift chart

# Gains/Lift Chart - Graph inc
# (X-axis - % of All cases) vs (Y-axis - % of True Cases ) - Line Graph is the cummulative % of true cases by each cumulative decile/group

# Actual vs Predicted - Graph dec
# (X-axis - % of All cases) vs (Y-axis - Count of True Cases) - Line graph with 2 lines-one for actual and one for predicted, lines are not cumulative true cases but they are count of true cases wrt to that decile population

# Lift Chart - Graph dec
# Lift value of each decile - Cummulative Number of True cases / Cummulative Number of population
# (X-axis - % of population) vs (Y-axis - Lift value) - https://www.saedsayad.com/model_evaluation_c.htm

# K-S Kolomogorov Smirnov Chart - http://www.saedsayad.com/model_evaluation_c.htm
# K-S is a measure of the degree of separation between the positive and negative distributions
# Here the observations are ordered in increasing order of Predicted probability to create deciles
# For each decile - Calculate counts, cumulative counts, cumulative count% of 1s and 0s for each decile
# K-S value for each decile is the difference between Cumulative 1s % and cumulatiev 0s % - Max k-s value within all deciles is the final K-S stat


y - Target Variable
y_Pred - Predicted Target variable as category and NOT continuous prob value
y_pred_proba - Predicted Target variable as continuous prob value

# Confusion Matrix
from sklearn.metrics import confusion_matrix
labels = ['label1', 'label2'] # label1 & label2 should be the labels in the y
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

# Precision Recall curve - Visualization makes more sense when prediction are made using method='predict_proba'
# predict_proba can be used in precision_recall_curve function but not in recall_score,precision_score functions
precision, recall, thresholds = precision_recall_curve(y,y_pred_proba)
# create plot 
plt.plot(precision, recall, label='Precision-recall curve'+', recall='+str(round(recall_score(y, y_Pred).mean(),3))+', precision='+str(round(precision_score(y, y_Pred).mean(),3)))
_ = plt.xlabel('Precision')
_ = plt.ylabel('Recall')
_ = plt.title('Precision-recall curve')

# 
#
         


         
         
# Decile Plots and Kolmogorov Smirnov (KS) Statistic (KS chart)
# Concordant-Discordant ratio
        





