# CANNOT handle MISSING values
# CANNOT handle categorical data - Should be encoded using different encoding techniques
# Target Variable can be BinaryLabel/MultiLabel and Numeric/Character

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

# DecisionTreeClassifier(max_depth=1) is the default base estimator

## List of base_estimators that can be used in AdaBoost
# Code
import inspect
from sklearn.utils.testing import all_estimators
for name, clf in all_estimators(type_filter='classifier'):
    if 'sample_weight' in inspect.getargspec(clf().fit)[0]:
       print(name)
# If the classifier doesn't implement predict_proba, you will have to set AdaBoostClassifier parameter algorithm = 'SAMME'. 
# Any classifier that supports passing sample weights should work - Higher weights force the classifier to put more emphasis on these points.
# base_estimators
AdaBoostClassifier
BaggingClassifier
BernoulliNB
CalibratedClassifierCV
ComplementNB
DecisionTreeClassifier
ExtraTreeClassifier
ExtraTreesClassifier
GaussianNB
GradientBoostingClassifier
LinearSVC
LogisticRegression
LogisticRegressionCV
MultinomialNB
NuSVC
Perceptron
RandomForestClassifier
RidgeClassifier
RidgeClassifierCV
SGDClassifier
SVC

## To specify base_estimator
svc=SVC(probability=True, kernel='linear')
# Create adaboost classifer using svc
AdaBoostModel = AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)




