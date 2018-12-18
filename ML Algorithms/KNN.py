# https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/
# https://kevinzakka.github.io/2016/07/13/k-nearest-neighbor/
# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

from sklearn.neighbors import KNeighborsClassifier

# Feature scaling
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

# Model
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train) 

# To find optimal value of k
neighbors = list(range(1,50))
# empty list that will hold cv scores
cv_scores = []
# perform 10-fold cross validation
for k in myList:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
# changing to misclassification error
MSE = [1 - x for x in cv_scores]




