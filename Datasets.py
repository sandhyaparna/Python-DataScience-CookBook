### Credit Approval Data Set ###
https://archive.ics.uci.edu/ml/datasets/Credit+Approval

  ### Credit Approval Data Set ###
# Link - https://archive.ics.uci.edu/ml/datasets/Credit+Approval

# Save crx.data as csv format
# Import saved data file
CreditApproval = pd.read_csv("C:/Users/spashikanti/Desktop/Objectives/Python/Sample DataSets/CreditApproval.csv")

# Replace "?" with np.NAN
CreditApproval.A1 = np.where(CreditApproval.A1=='?',np.NAN,CreditApproval.A1)
CreditApproval.A2 = np.where(CreditApproval.A2=='?',np.NAN,CreditApproval.A2)
CreditApproval.A4 = np.where(CreditApproval.A4=='?',np.NAN,CreditApproval.A4)
CreditApproval.A5 = np.where(CreditApproval.A5=='?',np.NAN,CreditApproval.A5)
CreditApproval.A6 = np.where(CreditApproval.A6=='?',np.NAN,CreditApproval.A6)
CreditApproval.A7 = np.where(CreditApproval.A7=='?',np.NAN,CreditApproval.A7)
CreditApproval.A14 = np.where(CreditApproval.A14=='?',np.NAN,CreditApproval.A14)

# Change Data types - If a numeric variable has missing data it cannot be stored as int and should be stored as float
CreditApproval.A2 = CreditApproval.A2.astype('float')
CreditApproval.A14 = CreditApproval.A14.astype('float')

# Categorical Variable - Replace values within Target
CreditApproval['Target_Cat'] = np.where(CreditApproval.Target=='+',"Pos","Neg")
# Numeric Variable - Replace values within Target
CreditApproval['Target_Num'] = np.where(CreditApproval.Target=='+',1,0)

# Create new variables with no missing data 
# Missing data is present in both cont and cat vars

# For Categorical vars - Create a new category for missing data
CreditApproval['A1_NoMissing'] = np.where(pd.isnull(CreditApproval.A1),'Miss',CreditApproval.A1)
CreditApproval['A4_NoMissing'] = np.where(pd.isnull(CreditApproval.A4),'Miss',CreditApproval.A4)
CreditApproval['A5_NoMissing'] = np.where(pd.isnull(CreditApproval.A5),'Miss',CreditApproval.A5)
CreditApproval['A6_NoMissing'] = np.where(pd.isnull(CreditApproval.A6),'Miss',CreditApproval.A6)
CreditApproval['A7_NoMissing'] = np.where(pd.isnull(CreditApproval.A7),'Miss',CreditApproval.A7)

# For Continuous vars - Replace missing data with mean
CreditApproval['A2_NoMissing'] = CreditApproval['A2'].fillna(CreditApproval['A2'].mean())
CreditApproval['A14_NoMissing'] = CreditApproval['A14'].fillna(CreditApproval['A14'].mean())

# Convert A1,A4,A5,A6,A7,A9,A10,A12,A13 to numeric - Label encoding
CreditApproval['A1_Num'] = CreditApproval['A1'].map({'a':1, 'b':0})
CreditApproval['A4_Num'] = CreditApproval['A4'].map({'u':0, 'y':1, 'l':2})
CreditApproval['A5_Num'] = CreditApproval['A5'].map({'g':0, 'p':1, 'gg':2})
CreditApproval['A6_Num'] = CreditApproval['A6'].map({'c':0, 'd':1, 'cc':2, 'i':3, 'j':4, 'k':5, 'm':6, 'r':7, 'q':8, 'w':9, 'x':10, 'e':11, 'aa':12, 'ff':13})
CreditApproval['A7_Num'] = CreditApproval['A7'].map({'v':0, 'h':1, 'bb':2, 'j':3, 'n':4, 'z':5, 'dd':6, 'ff':7, 'o':8})
CreditApproval['A9_Num'] = CreditApproval['A9'].map({'t':1, 'f':0})
CreditApproval['A10_Num'] = CreditApproval['A10'].map({'t':1, 'f':0})
CreditApproval['A12_Num'] = CreditApproval['A12'].map({'t':1, 'f':0})
CreditApproval['A13_Num'] = CreditApproval['A13'].map({'g':0, 'p':1, 's':2})

# Encode NoMissing Char Variables - Use One-hot encoding
Char_NoMissing_OneHotEncoder = pd.get_dummies(CreditApproval[['A1_NoMissing','A4_NoMissing','A5_NoMissing','A6_NoMissing','A7_NoMissing','A9','A10','A12','A13']],drop_first=True)

# Encode NoMissing Char Variables - Use LabelEncoder
from sklearn import *
from sklearn.preprocessing import *
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

CharFeatures =  ['A1_NoMissing', 'A4_NoMissing', 'A5_NoMissing', 'A6_NoMissing', 'A7_NoMissing','A9','A10','A12','A13']
Char_NoMissing_LabelEncoder = MultiColumnLabelEncoder(CharFeatures).fit_transform(CreditApproval[CharFeatures]) 
Char_NoMissing_LabelEncoder = Char_NoMissing_LabelEncoder.add_suffix('_LabelEncoder')

# Join Encoder data sets - 79 Variables
CreditApproval = pd.concat([CreditApproval,Char_NoMissing_OneHotEncoder,Char_NoMissing_LabelEncoder])

# Create a multiclass Target variable using kmeans (3 classes)
# extract only Continuous variables for k-means - Doesn't accept missing data
from sklearn.cluster import KMeans
Cont = CreditApproval[['A2_NoMissing','A3','A8','A11','A14_NoMissing','A15']]
kmeans = KMeans(n_clusters=20, random_state=0).fit(Cont) #Created 20 clusters as 3 clusters doesnt give 3 well defined clusters

CreditApproval = CreditApproval.assign(Clusters=kmeans.labels_)
CreditApproval['Clusters_Cat'] = np.where(CreditApproval.Clusters==12,"Clus_1",
                                     np.where(CreditApproval.Clusters==0,"Clus_2","Clus_3"))
CreditApproval['Clusters_Num'] = np.where(CreditApproval.Clusters==12,1,
                                     np.where(CreditApproval.Clusters==0,2,3))                                    

# save
CreditApproval.to_pickle("C:/Users/spashikanti/Desktop/Objectives/Python/Sample DataSets/CreditApproval.pkl")








