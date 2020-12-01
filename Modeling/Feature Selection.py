### Weight of evidence and Information Value using Python
# https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb
# https://github.com/Sundar0989/WOE-and-IV/blob/master/WOE_IV.ipynb
# https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/

### Feature selection on Categorical vars:
# chi-sq: Between Dependent and Independent vars
fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X_train, y_train)
# X_train_fs = fs.transform(X_train)
# X_test_fs = fs.transform(X_test)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

# Mutual information from the field of information theory is the application of information gain (typically used in the construction of decision trees) to feature selection.
# Mutual information is calculated between two variables and measures the reduction in uncertainty for one variable given a known value of the other variable.
# uses score_func=mutual_info_classif in SelectKBest
fs = SelectKBest(score_func=mutual_info_classif, k='all')






