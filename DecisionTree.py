# CANNOT handle MISSING values
# CANNOT handle categorical data - Should be encoded using different encoding techniques
# Target Variable can be BinaryLabel/MultiLabel and Numeric/Character

# scoring="roc_auc" works only when Target Var is BinaryNumeric
# But If no 'scoring' parameter is mentioned, it produces roc-auc by default for any type of Target var

# cross_val_predict function should have method='predict' & not method='predict_proba'
# method='predict_proba' - Generates array with 2 columns for Binary, 3 columns for Multi (columns correspond to the classes in sorted order)
# method='predict' - Generates array with column and different predicted categories(Original Categories as inputed for training)
















