# https://www.kdnuggets.com/2016/08/include-high-cardinality-attributes-predictive-model.html
  
  
# Binary Features
  # Map Values
# Low & High(a lot of unique values) Cardinality Nominal Features
  # Nominal - Hashing, LeaveOneOut, Target encoding (For regression - Target & LeaveOneOut won't work well)
  # High Cardinal Data - Target (Mean of DV), LeaveOneOut, WeightOfEvidence, 
  
# Low & High(a lot of unique values) Cardinality Ordinal Features
  # Oridinal, Binary, OneHot, LeaveOneOut, Target encoding
# Cyclical Features - Day, week, month, year 
  # sin-cos features

# Label & Freq encoding works well for tree based methods. Tree methods slow down in OnehotEncoding
# On-Hot encoding works well for Non-Tree based models

  

### Character variables encoding ###
Df_X is data frame with features
# http://pbpython.com/categorical-encoding.html
# https://github.com/scikit-learn-contrib/categorical-encoding
# http://contrib.scikit-learn.org/categorical-encoding/
# https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159
# https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02?_branch_match_id=568145403917380761
# https://www.kaggle.com/subinium/11-categorical-encoders-and-benchmark#1.-Label-Encoder-(LE),-Ordinary-Encoder(OE)
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
label_encoder = preprocessing.LabelEncoder() #Alphabetical order based
Df_Var_array = label_encoder.fit_transform(Df_X['Var'])
- Pandas.factorize is used for Label encoding in order of appearance

# b) MultiColumnLabelEncoder - should be used only on categorical vars
# i) 
lencoder = ['ord_5', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'day', 'month']
test['target'] = 'test'
df = pd.concat([train, test], axis=0, sort=False )
for feat in lencoder:
    lbl_enc = preprocessing.LabelEncoder()
    df[feat] = lbl_enc.fit_transform(df[feat].values)
# ii) It encodes integer variables also - so only char variables should be mentioned
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
# %%time
# this method didn't work because of RAM memory. 
# HE_encoder = HelmertEncoder(feature_list)
# train_he = HE_encoder.fit_transform(train[feature_list], target)
# test_he = HE_encoder.transform(test[feature_list])

# 6. Freq Encoding
# Freq of a particular category in the whole data
encoding  = Df.groupby('Var').size()
encoding = encoding/len(Df)
Df['NewVar'] = Df.Var.map(encoding)

from scipy.stats import rankdata

# 7. Mean Encoding
#   Select a categorical variable you would like to transform
#   Group by the categorical variable and obtain aggregated sum over “Target” variable. (total number of 1’s for each category in ‘Temperature’)
#   Group by the categorical variable and obtain aggregated count over “Target” variable
#   Divide the step 2 / step 3 results and join it back with the train.

# 8. Weight of Evidence Encoding
%%time
WOE_encoder = WOEEncoder()
train_woe = WOE_encoder.fit_transform(train[feature_list], target)
test_woe = WOE_encoder.transform(test[feature_list])

# 9. Supervise Ratio

# 10. Ordinal
# a) Values are sorted and dictionary is created - works weel with Alphabets
grade = sorted(list(set(X_train['grade'].values)))
grade = dict(zip(grade, [x+1 for x in range(len(grade))]))
X_train.loc[:, 'grade'] = X_train['grade'].apply(lambda x: grade[x]).astype(int)
X_test.loc[:, 'grade'] = X_test['grade'].apply(lambda x: grade[x]).astype(int)
# b) The first unique value in your column becomes 1, the second becomes 2, the third becomes 3. So, mapping is best for correct & accurate mapping
ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',
                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)
ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)

# 11. Hashing
# Default is 8 columns

# 12. M-Estimate Encoder
%%time
MEE_encoder = MEstimateEncoder()
train_mee = MEE_encoder.fit_transform(train[feature_list], target)
test_mee = MEE_encoder.transform(test[feature_list])

# 13. Target Encoder
%%time
TE_encoder = TargetEncoder()
train_te = TE_encoder.fit_transform(train[feature_list], target)
test_te = TE_encoder.transform(test[feature_list])

# 14. James-Stein Encoder
%%time
JSE_encoder = JamesSteinEncoder()
train_jse = JSE_encoder.fit_transform(train[feature_list], target)
test_jse = JSE_encoder.transform(test[feature_list])

# 15. Leave-one-out Encoder (LOO or LOOE)
%%time
LOOE_encoder = LeaveOneOutEncoder()
train_looe = LOOE_encoder.fit_transform(train[feature_list], target)
test_looe = LOOE_encoder.transform(test[feature_list])

# 16. Catboost Encoder
%%time
CBE_encoder = CatBoostEncoder()
train_cbe = CBE_encoder.fit_transform(train[feature_list], target)
test_cbe = CBE_encoder.transform(test[feature_list])

# 17. Sin-cos encoding - for cyclical features
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
data = encode(data, 'month', 12)
data = encode(data, 'hr', 23)
data = encode(data, 'year', 365)

#### Apply different encoding techniques simultaneously -https://www.kaggle.com/subinium/11-categorical-encoders-and-benchmark#1.-Label-Encoder-(LE),-Ordinary-Encoder(OE)
%%time
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression

encoder_list = [ OrdinalEncoder(), WOEEncoder(), TargetEncoder(), MEstimateEncoder(), JamesSteinEncoder(), LeaveOneOutEncoder() ,CatBoostEncoder()]

X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=97)

for encoder in encoder_list:
    print("Test {} : ".format(str(encoder).split('(')[0]), end=" ")
    train_enc = encoder.fit_transform(X_train[feature_list], y_train)
    #test_enc = encoder.transform(test[feature_list])
    val_enc = encoder.transform(X_val[feature_list])
    lr = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
    lr.fit(train_enc, y_train)
    lr_pred = lr.predict_proba(val_enc)[:, 1]
    score = auc(y_val, lr_pred)
    print("score: ", score)
    del train_enc
    del val_enc
    gc.collect()







