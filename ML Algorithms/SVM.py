# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769

# Parameters
# Kernel: 'linear' will use a linear hyperplane. ‘rbf’ and ‘poly’ uses a non linear hyper-plane.
# Go for linear kernel if you have large number of features (>1000) because it is more likely that the data is linearly separable in high dimensional space.
# gamma: Used for non linear hyper-planes. Higher the gamma value it tries to exactly fit the training data set. Can lead to over-fitting
# C: Penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying the training points correctly.
# Increasing C values may lead to overfitting the training data.
# degree: Used when kernel is set to ‘poly’. It’s basically the degree of the polynomial used to find the hyperplane to split the data.

from sklearn import svm
# Create SVM classification object 
model = svm.svc(kernel='linear', c=1, gamma=1) 
model.fit(X, y)
model.score(X, y)




