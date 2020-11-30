# erik@interviewqs.com

### Question 1 - Fradulent retail accounts
# store_account table, columns are: store_id, date, status, revenue
# Here's some more information about the table:
# The granularity of the table is store_id and day
# Assume “close” and “fraud” are permanent labels
# Active = daily revenue > 0
# Accounts get labeled by Shopify as fraudulent and they no longer can sell product
# Every day of the table has every store_id that has ever been used by Shopify
# write code using Python (Pandas library) to show what percent of active stores were fraudulent by day.
# Some clarifications:
# We want one value for each day in the month.
# A store can be fraudulent and active on same day. E.g. they could generate revenue until 10AM, then be flagged as fradulent from 10AM onward.
### Answer
Active_stores =  store_account[store_account.revenue>0]
Active_stores['Fraud_Identifier'] = np.where(Active_stores.status=="fraud",1,0)
Active_stores_Fraud = pd.DataFrame(Active_stores.groupby('date').agg({'Fraud_Identifier':['sum','count']})).reset_index()
Active_stores_Fraud['Fraud_Percent'] = Active_stores_Fraud.sum/Active_stores_Fraud.count

### Question
# Calculating a moving average using Python. sliding window average. Get min and max average of the sliding windows
# You are given a list of numbers J and a single number p. Write a function to return the minimum and maximum averages of the sequences of p numbers in J
### Answer
import numpy as np
J = [4, 4, 4, 9, 10, 11, 12]
p = 3
J = np.sort(J,axis=None)
n = len(J)
min_max_array = []
for i in [0,n-p]:
  min_max = np.mean(J[i:][0:p])
  min_max_array.append(min_max)
min_max_array









