############ Binning ############
# Binning helps reducing the noise or non-linearity. Allows easy identification of outliers, invalid and missing values of numerical variables 
# Do not use target class info during Binning 

# http://www.saedsayad.com/binning.htm

### Equal Width Binning
# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.cut.html
# k is number of bins to be created
# Width of each bin = (Max-Min)/k
Df['Var_Bin'] = pd.cut(Df['Num_Var'],k) 
# W is width & Var_Bin is a numeric column with bin number, starts from 0. for eg: Age 0-9 for Width=10 will have bin value 0 
Df['Var_Bin'] = np.array(np.floor(np.array(Df['Num_Var']) / W))

### User Defined Binning ###
# DONT use square brackets while creating bins-even if (), it is actually equivalent to (]
bins = pd.IntervalIndex.from_tuples([(10,20), (20, 30), (30,40), (40, 50),(50,60), (60, 70),(70,80), (80, 90)])
bins_label = [1, 2, 3, 4, 5, 6]
Df['Var_Bin'] = pd.cut(Df['Num_Var'], bins)
Df['Var_Bin'] = pd.cut(np.array(Df['Num_Var']), bins)
Df['Var_BinLabel'] = pd.cut(np.array(Df['Num_Var']), bins=bins, labels=bins_label)

### Equal Freq binning
# https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.qcut.html
# q is number of quantiles or array of quantiles
# q is 10 for deciles, 4 for quantiles etc; array of quantiles:[0, .25, .5, .75, 1.] for quartiles
Df['Var_Bin'] = pd.qcut(Df['Num_Var'],q) 

### Entropy Based Binning
# http://www.saedsayad.com/unsupervised_binning.htm
# https://github.com/paulbrodersen/entropy_based_binning


############ Box-cox Transformation ############
import spstats
Num_Var = np.array(Df['Num_Var'])
Num_Var_clean = Num_Var[~np.isnan(Num_Var)]
l, opt_lambda = spstats.boxcox(Num_Var_clean)
print('Optimal lambda value:', opt_lambda)
# Transformation as a new var
l, opt_lambda = spstats.boxcox(Df['Num_Var'])
Df['Num_Var_boxcox_lambda_opt'] = spstats.boxcox(Df['Num_Var'], lmbda=opt_lambda)


############ Log Transformation ############
Df['Num_Var_log'] = np.log((1+ Df['Num_Var']))


############ Distance calculations on Globe ############
# geopy module in Python, Openstreetmap
# Haversine Distance Between the Two Lat/Lons:
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h
# Manhattan Distance Between the two Lat/Lons:
def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b    
# Bearing Between the two Lat/Lons:
def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
# Center Latitude and Longitude between Pickup and Dropoff:
train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2

############# Pre-processing ##############
### Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
 standardized_X_test = scaler.transform(X_test)

### Normalization
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
    
### Binarization
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)

### Impute missing values
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0, strategy='mean', axis=0)
imp.fit_transform(X_train)

### Generating Polynomial Features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(5)
poly.fit_transform(X) 


############ Web Data ############
# https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-6-feature-engineering-and-feature-selection-8b94f870706a
# 
n : ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/56.0.2924.76 Chrome/ 
...: 56.0.2924.76 Safari/537.36'

In : import user_agents

In : ua = user_agents.parse(ua)

In : ua.is_bot 
Out: False

In : ua.is_mobile 
Out: False

In : ua.is_pc 
Out: True

In : ua.os.family 
Out: 'Ubuntu'

In : ua.os.version 
Out: ()

In : ua.browser.family 
Out: 'Chromium'

In : ua.os.version 
Out: ()

In : ua.browser.version 
Out: (56, 0, 2924)

    
    
    
    
    
    
