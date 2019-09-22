# Python libraries for model interpretation
https://www.kdnuggets.com/2019/09/python-libraries-interpretable-machine-learning.html

### ELI5 ###
https://github.com/TeamHG-Memex/eli5

### Skater ###
https://github.com/datascienceinc/Skater

### LIME ###
https://github.com/marcotcr/lime
Python code with clear explanation on how to implement    
https://pythondata.com/local-interpretable-model-agnostic-explanations-lime-python/
# Train data on Random forest
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)
train, test, labels_train, labels_test = train_test_split(boston.data, boston.target, train_size=0.80)
rf.fit(train, labels_train)
# Extract Categorical features
categorical_features = np.argwhere(
    np.array([len(set(boston.data[:,x]))
    for x in range(boston.data.shape[1])]) <= 10).flatten()
# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(train, 
                                                   feature_names=boston.feature_names, 
                                                   class_names=['price'], 
                                                   categorical_features=categorical_features, 
                                                   verbose=True, mode='regression')
# Now, we can grab one of our test values and check out our prediction(s)
i = 100 # 100th test observation is used
exp = explainer.explain_instance(test[i], rf.predict, num_features=5)
exp.show_in_notebook(show_table=True)
    
    
### SHAP ###
* Cannot contain Missing values
https://github.com/slundberg/shap
http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine.html
https://christophm.github.io/interpretable-ml-book/shapley.html#
Comparision of interpreting features using diff methods - http://savvastjortjoglou.com/intrepretable-machine-learning-nfl-combine.html
SHAP on Finance data - https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
SHAP on Health data - https://slundberg.github.io/shap/notebooks/NHANES%20I%20Survival%20Model.html
# SHAP(SHapley Additive exPlanations). SHAP assigns each feature an importance value for a particular prediction. 
# Help measure the impact of the features on the predictions
# Tree SHAP method is mathematically equivalent to averaging differences in predictions over all possible orderings of the features, rather than just the ordering specified by their position in the tree.
import shap
# Below code produces shap value for each column of all observations 
shap_values = shap.TreeExplainer(XGBoostModel).shap_values(X)
# Bar chart of feature importance
shap.summary_plot(shap_values, X, plot_type="bar")
# Chart of feature importance for each observation
shap.summary_plot(shap_values, X) 
# Shap values of a particular feature vs Actual feature values - SHAP dependence plot show how the model output varies by feauture value
# The feature(second feature) used for coloring is automatically chosen to highlight what might be driving these interactions.
shap.dependence_plot("Var1",shap_values, X)
# SHAP dependence plot of all var names in X
for var in X.columns:
    shap.dependence_plot(var, shap_values, X)   
# SHAP Interaction Value Summary Plot
shap_interaction_values = shap.TreeExplainer(XGBoostModel).shap_interaction_values(X)
shap.summary_plot(shap_interaction_values, X)
# To choose 2nd feature by your choice and no automatic
shap.dependence_plot(("Var1", "Var1"),shap_interaction_values, X) #Var1 only
shap.dependence_plot(("Var1", "Var2"),shap_interaction_values, X) #choose Var2 instead of automatic choosing















