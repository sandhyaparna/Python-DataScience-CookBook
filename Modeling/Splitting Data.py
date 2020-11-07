# Split data into Train and Test by CustomerID. Each CustomID may have multiple records
CustomerIDs = pd.unique(CustomerRatings.CustomerID)
CustomerIDs_Train = random.sample(list(CustomerIDs),np.ceil(len(CustomerIDs)*0.8).astype(int))
CustomerIDs_Test = set(CustomerIDs).difference(set(CustomerIDs_Train))

CustomerRatings_Train = CustomerRatings[CustomerRatings.CustomerID.isin(CustomerIDs_Train)]
CustomerRatings_Test = CustomerRatings[CustomerRatings.CustomerID.isin(CustomerIDs_Test)]








