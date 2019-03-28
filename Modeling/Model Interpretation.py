### Skater ###
https://github.com/datascienceinc/Skater

### LIME ###
https://github.com/marcotcr/lime

### SHAP ###
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















