# When doing oversampling or undersampling, validation and train data for cross validation needs to be prepared manually as oversampling should not be done on Validation sets
# https://kiwidamien.github.io/how-to-do-cross-validation-when-upsampling-data.html
  
# Split data into Train and Test by CustomerID. Each CustomID may have multiple records
CustomerIDs = pd.unique(CustomerRatings.CustomerID)
CustomerIDs_Train = random.sample(list(CustomerIDs),np.ceil(len(CustomerIDs)*0.8).astype(int))
CustomerIDs_Test = set(CustomerIDs).difference(set(CustomerIDs_Train))

CustomerRatings_Train = CustomerRatings[CustomerRatings.CustomerID.isin(CustomerIDs_Train)]
CustomerRatings_Test = CustomerRatings[CustomerRatings.CustomerID.isin(CustomerIDs_Test)]

# Split first 80% rows into Train and remaining 20% rows into Test
Train = Df.iloc[:int(Df.shape[0]*0.80)] #Extracting first 80%
Test = Df.iloc[int(Df.shape[0]*0.80):] # Exracting remaining






