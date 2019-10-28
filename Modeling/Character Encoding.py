# https://www.kdnuggets.com/2016/08/include-high-cardinality-attributes-predictive-model.html
  
  
# Binary Features
  # Map Values
# Low & High(a lot of unique values) Cardinality Nominal Features
  # Nominal - Hashing, LeaveOneOut, Target encoding (For regression - Target & LeaveOneOut won't work well)
  # High Cardinal Data - Target (Mean of DV), LeaveOneOut, WeightOfEvidence, 
  
# Low & High(a lot of unique values) Cardinality Ordinal Features
  # Oridinal, Binary, OneHot, LeaveOneOut, Target encoding
# Cyclical Features 
  

  
  
  
### Character variables encoding ###
Df_X is data frame with features
# http://pbpython.com/categorical-encoding.html
# https://github.com/scikit-learn-contrib/categorical-encoding
# http://contrib.scikit-learn.org/categorical-encoding/
# https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159
# https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02?_branch_match_id=568145403917380761

# When to use what kind of encoding - https://miro.medium.com/max/2100/0*NBVi7M3sGyiUSyd5.png

# 1. Replace/Rename/Map Values of a variable (CAN be USED for MISSING Vars)
# a)
Char_Codes = {"Char_Var1": {"Value1": New_Vaue1, "Value2": New_Vaue2},
              "Char_Var2": {"Value1": New_Vaue1, "Value2": New_Vaue2, "Value3": New_Vaue3, "Value4": New_Vaue4 }}
Df.replace(Char_Codes, inplace=True)
# b) Can be used when there are missing values - As Manula encoding doesnt change misisng values
Df['Var'] = Df['Var'].map({'Value1':New_Vaue1, 'Value2':New_Vaue2, 'Value3':New_Vaue3})

# 1. Eg- convert Trure or False, Yes or No to Binary
bin_dict = { 'T':1,'F':0,'Y':1,'N':0}
test['bin_3'] = test['bin_3'].map(bin_dict)  

# 2. Label encoding - Using Categories (CANNOT be USED for MISSING Vars)
# a) Single variable encoding
# i) Label encoding - Datatype of variable should be converted to character
Df_X['Var'] = Df_X['Var'].astype('category')
Df_X['Var'] = Df_X['Var'].cat.codes
# ii) Label encoding - Initialize label encoder - Doesnt work if there is missing data
label_encoder = preprocessing.LabelEncoder()
Df_Var_array = label_encoder.fit_transform(Df_X['Var'])

# b) MultiColumnLabelEncoder - should be used only on categorical vars
# It encodes integer variables also - so only char variables should be mentioned
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
class MultiColumnLabelEncoder:
    
    def __init__(self, columns = None):
        self.columns = columns # list of column to encode
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        
        output = X.copy()
        
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        
        return output
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
-
CharFeatures = list(Df_X.select_dtypes(include=['object']))
Df_LabelEncoded = MultiColumnLabelEncoder(CharFeatures).fit_transform(Df_X[CharFeatures])        
    
# 3. One-hot encoding - Replace existing variable values with new encoding - (CANNOT be USED for MISSING Vars)
# a) Cannot use this when there are missing values - For eg, if there are 3 uniques in a var and a few missing values, a=10, b=30, c=5, np.NAN =5, Only 2 columns will be created Var_a, var_b i.e if (Var_a=0 & Var_b=0) represents both c cat as well as missing data  
Df_OneHotEncoded = pd.get_dummies(Df_X,drop_first=True) 
# b) LabelBinarizer is also one-hot encoding - But only single variable
Label_Binarizer = LabelBinarizer()
Df_Var = Label_Binarizer.fit_transform(Df_X['Var'])
Df_Var = pd.DataFrame(Df_Var, columns=Label_Binarizer.classes_)

# 4. BinaryEncoders using category_encoders *******
# First the categories are encoded as ordinal, then those integers are converted into binary code, then the digits from that binary string are split into separate columns.  This encodes the data in fewer dimensions that one-hot
# Performs consistently well for actual categorical vars
import category_encoders as ce
Df_y = Df['Target_Var']
Df_X is data frame with features
CharFeatures = list(Df_X.select_dtypes(include=['object']))
Df_X_BinaryEncoder = ce.BinaryEncoder(cols=CharFeatures).fit(Df_X, Df_y)
Df_X_BinaryEncoder = Bank_X_BinaryEncoder.transform(Df_X)

# 5. Helmert Encoding -

# 6. Freq Encoding
# Freq of a particular category in the whole data

# 7. Mean Encoding
#   Select a categorical variable you would like to transform
#   Group by the categorical variable and obtain aggregated sum over “Target” variable. (total number of 1’s for each category in ‘Temperature’)
#   Group by the categorical variable and obtain aggregated count over “Target” variable
#   Divide the step 2 / step 3 results and join it back with the train.

# 8. Weight of Evidence Encoding

# 9. Supervise Ratio

# 10. Ordinal
# The first unique value in your column becomes 1, the second becomes 2, the third becomes 3. So, mapping is best for correct & accurate mapping
ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',
                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)
ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)

# 11. Hashing
# Default is 8 columns

