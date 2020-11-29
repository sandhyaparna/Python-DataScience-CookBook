# https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.over_sampling
# Below consists of SMOTE, SMOTENC, Borderline-SMOTE, SVM SMOTE, ADASYN
# https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5

# SMOTE
from imblearn.over_sampling import SMOTE,ADASYN

sm = SMOTE(random_state=42)
X_SMOTE,y_SMOTE = sm.fit_resample(X,y)

ada = ADASYN(random_state=42)
X_ADASYN,y_ADASYN = ada.fit_resample(X,y)

sum(y_SMOTE==1) #Gives Target=1 values in the smote data
sum(y_train_SMOTE==0) #Gives Target=0 values in the smote data
X_SMOTE.shape # 


https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
### Weight Balancing
import keras
class_weight = {"buy": 0.75,  #Positive class is given more weight
                "don't buy": 0.25}
model.fit(X_train, Y_train, epochs=10, batch_size=32, class_weight=class_weight)

### Focal loss
import keras
from keras import backend as K
import tensorflow as tf
# Define our custom loss function
def focal_loss(y_true, y_pred):
    gamma = 2.0, alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
# Compile our model
adam = Adam(lr=0.0001)
model.compile(loss=[focal_loss], metrics=["accuracy"], optimizer=adam) 


# Tomek Links
## In the code below, we’ll use ratio='majority' to resample the majority class.
# import library
from imblearn.under_sampling import TomekLinks
tl = RandomOverSampler(sampling_strategy='majority')
# fit predictor and target variable
x_tl, y_tl = ros.fit_resample(x, y)
print('Original dataset shape', Counter(y))
print('Resample dataset shape', Counter(y_ros))






