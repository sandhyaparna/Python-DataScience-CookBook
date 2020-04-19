# https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.over_sampling
# SMOTE
from imblearn.over_sampling import SMOTE,ADASYN

sm = SMOTE(random_state=42)
X_SMOTE,y_SMOTE = sm.fit_resample(X,y)

ada = ADASYN(random_state=42)
X_ADASYN,y_ADASYN = ada.fit_resample(X,y)

sum(y_SMOTE==1) #Gives Target=1 values in the smote data
sum(y_train_SMOTE==0) #Gives Target=0 values in the smote data
X_SMOTE.shape # 


### Weight Balancing
import keras
class_weight = {"buy": 0.75,  #Positive class is given more weight
                "don't buy": 0.25}
model.fit(X_train, Y_train, epochs=10, batch_size=32, class_weight=class_weight)










