# Problem Statement
# A Patient is given different visit IDs every time they visit a hospital. 
# Some times even during the same visit of a patient they give different patient IDs.
# So based on Arrival and Departure dates we should identify 'Actual Arrival' and 'Actual Departure' for each visit of a patient
# We assume that if current departure and next arrival time difference is less than or equal to 48hrs, they belong to the same visit
# Actual Arrival & Actual Departure dates for below eg are:
  #     Arrival  Departure  
  # 2018-01-01 2018-01-05                      
  # 2018-01-10 2018-01-13                      
  # 2018-02-03 2018-02-27                      
  # 2018-03-02 2018-03-16                   
  # 2018-03-20 2018-03-23                     
  # 2018-03-26 2018-03-31                     
  # 2018-04-05 2018-04-18  

# Data - One Patient's different visits
UniqueID	Arrival	Departure
1	1-Jan	5-Jan
2	10-Jan	13-Jan
3	3-Feb	8-Feb
4	3-Feb	12-Feb
5	3-Feb	4-Feb
6	3-Feb	7-Feb
7	3-Feb	6-Feb
8	4-Feb	16-Feb
9	4-Feb	6-Feb
10	4-Feb	14-Feb
11	7-Feb	13-Feb
12	10-Feb	15-Feb
13	18-Feb	19-Feb
14	19-Feb	27-Feb
15	20-Feb	23-Feb
16	22-Feb	25-Feb
17	23-Feb	26-Feb
18	2-Mar	8-Mar
19	3-Mar	16-Mar
20	11-Mar	13-Mar
21	15-Mar	15-Mar
22	20-Mar	23-Mar
23	26-Mar	30-Mar
24	30-Mar	28-Mar
25	30-Mar	31-Mar
26	5-Apr	18-Apr

### Logic ###
# Patients cohort includes any visit that has atleast one reading of Vitals or labs
# Order the data by ascending order of Arrival & Departure
# Now try to remove duplicate Arrivals - By Calculating max Departure for each Arrival
# Now there are no duplicates in Arrival and data is sorted by Arrivals
# Now check if there is atleast one observation where Next_Arrival is within (current Arrival & Current Departure+2Days) except last observation/visit
# If there is atleast one observation like discussed above, implies the visits should be still combined
# So, to combine visits
# We need to keep updating Departure date of each Arrival - i.e for current visit, if Arrival is 3rd Feb and Dep is 14feb
  # If next Arrival Arrival is 5feb and 27 feb - current visit Dep should be updated from 14th feb to 27 feb..so on
# Self join Arrival & Dep to itself uisng Patient MRN
# Create 2 Vars
  #1. Dep_1 = If Arrival_y is between (Arrival_x & Departure_x+2Days) then Max(Dep_x,Dep_y) if not Dep_x
  #2. Dep_2 = If Arrival_x is between (Arrival_y & Departure_y+2Days) then Max(Dep_x,Dep_y) if not Dep_x - This is because previous visit ARrival may actually have max Dep date for the actual visit
# Dep_1 is at data in the self join where Arrival_y>ARrival_x 
  # y is checked within x - to find max Dep date
# Dep_2 is aimed at data in the self join where Arrival_y<Arrival_x - even if Arrivaly<Arrival_x - actual Dep might be here, happens where feb3-feb15th is previous visit and current visit is feb4th-feb10. we need to get feb15-so this var is created
  # X is checked with y dates - to find max Dep date
# So we capture an actual Departure from Arrival looking at next visit as well as previous visits
# Within each Arrival_x - choose the max date of departure among both Dep_1 & Dep_2
  # So concatenate dates row wise - (Arrival_x,Dep_1) and (Arrival_x,Dep-2) - Rename Dep_1 & Dep_2 to Departure so as to do concatenation
# Now for each Arrival we get max Departure date
# Now we remove Duplicate Arrivals - Because 3rd feb-give 27th feb and 4th feb gives 27 feb, 7th feb - gives 27 feb.
# But we only want 3rd feb to 27th feb
# So, we remove Duplicate Arrivals by Calculating min(Arrival) within each Departure
# Now for each Departure we get min Arrival date
# Repeat from step 4 till previous step untill there are no observations where Next_Arrival is within (current Arrival & Current Departure+2Days) except last observation/visit


#### MRN Sample Check code ####
MRNSample_Visits1 =  pd.read_csv("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/MRNSample_Visits1.csv",
                 parse_dates = ['Arrival','Departure']) # To import variable in datetime format default format is '%d%b%Y:%H:%M:%S.%f')

MRNSample_Visits1=MRNSample_Visits1[['Arrival','Departure']]
 
# Sort data by Arrival,Departure
MRNSample_Visits1 = MRNSample_Visits1.sort_values(by=['Arrival','Departure'])

# Remove cases where Departure is before Arrival

# Remove duplicate Arrivals - within each Arrival pick max Departure
import pip
from pandasql import *
import pandasql as ps
query = """select Arrival,max(Departure) as Departure from MRNSample_Visits1 group by Arrival"""
MRNSample_Visits1 = ps.sqldf(query, locals())

# Convert Arrival,Departure to datetime
import datetime
MRNSample_Visits1['Arrival'] =  pd.to_datetime(MRNSample_Visits1['Arrival'], format='%Y-%m-%d %H:%M:%S.%f')
MRNSample_Visits1['Departure'] =  pd.to_datetime(MRNSample_Visits1['Departure'], format='%Y-%m-%d %H:%M:%S.%f')

# Check if Next_Arrival is greater than Departure+2Days
MRNSample_Visits1['Next_Arrival_Dep_Gre2'] = np.where((MRNSample_Visits1['Arrival'].shift(-1))>((MRNSample_Visits1['Departure']) + pd.Timedelta(days=2)),1,0)

# Check if all values of Next_Arrival_Dep_Gre2 are 1except last obs within a MRN

#### 1st Iteration,2nd,3

# Create MRN 
MRNSample_Visits1 = MRNSample_Visits1.assign(MRN=1)

# Self join data using MRN
MRNSample_Visits1 = MRNSample_Visits1[['MRN','Arrival','Departure']].merge(MRNSample_Visits1[['MRN','Arrival','Departure']], how='inner',on='MRN')

# Since there are no duplicate Arrivals
# Delete obs where Arrival_y<Arrival_x and not Arrival_y<=Arrival_x because id last patientID is unique to itself - it goes away
# MRNSample_Visits1 = MRNSample_Visits1[(MRNSample_Visits1.Arrival_y>=MRNSample_Visits1.Arrival_x)]

# Check if time difference between A_y & A_x is >=0 & A_y &D_x<=2days if yes then Ay=Ax and Dy is max(Dx,Dy)
MRNSample_Visits1['Departure_New'] = np.where(( ((MRNSample_Visits1['Arrival_y'])<=((MRNSample_Visits1['Departure_x'])+(pd.Timedelta(days=2)))) & ((MRNSample_Visits1.Arrival_y)>=(MRNSample_Visits1.Arrival_x)) ), 
                                                np.maximum(MRNSample_Visits1['Departure_x'],MRNSample_Visits1['Departure_y']),
                                                MRNSample_Visits1['Departure_x'])

MRNSample_Visits1['Departure_New1'] = np.where(( ((MRNSample_Visits1['Arrival_x'])<=((MRNSample_Visits1['Departure_y'])+(pd.Timedelta(days=2)))) & ((MRNSample_Visits1.Arrival_x)>=(MRNSample_Visits1.Arrival_y)) ), 
                                                np.maximum(MRNSample_Visits1['Departure_x'],MRNSample_Visits1['Departure_y']),
                                                MRNSample_Visits1['Departure_x'])

# Extract Arrival_x and Departure_New
MRNSample_Visits1a = MRNSample_Visits1[['Arrival_x','Departure_New']].drop_duplicates()
MRNSample_Visits1a = MRNSample_Visits1a.rename(columns={"Departure_New":"Departure"})

MRNSample_Visits1b = MRNSample_Visits1[['Arrival_x','Departure_New1']].drop_duplicates()
MRNSample_Visits1b = MRNSample_Visits1b.rename(columns={"Departure_New1":"Departure"})

MRNSample_Visits1 = pd.concat([MRNSample_Visits1a,MRNSample_Visits1b])

# Remove duplicate Arrivals - within each Arrival pick max Departure
import pip
from pandasql import *
import pandasql as ps
query = """select Arrival_x as Arrival,max(Departure) as Departure from MRNSample_Visits1 group by Arrival_x"""
MRNSample_Visits1 = ps.sqldf(query, locals())

# Convert Arrival,Departure to datetime
import datetime
MRNSample_Visits1['Arrival'] =  pd.to_datetime(MRNSample_Visits1['Arrival'], format='%Y-%m-%d %H:%M:%S.%f')
MRNSample_Visits1['Departure'] =  pd.to_datetime(MRNSample_Visits1['Departure'], format='%Y-%m-%d %H:%M:%S.%f')

# Remove duplicate Arrivals - within each Departure pick min Arrival
import pip
from pandasql import *
import pandasql as ps
query = """select Departure,min(Arrival) as Arrival from MRNSample_Visits1 group by Departure"""
MRNSample_Visits1 = ps.sqldf(query, locals()) 

# Convert Arrival,Departure to datetime
import datetime
MRNSample_Visits1['Arrival'] =  pd.to_datetime(MRNSample_Visits1['Arrival'], format='%Y-%m-%d %H:%M:%S.%f')
MRNSample_Visits1['Departure'] =  pd.to_datetime(MRNSample_Visits1['Departure'], format='%Y-%m-%d %H:%M:%S.%f')

# Sort data by Arrival
MRNSample_Visits1 = MRNSample_Visits1.sort_values(by=['Arrival'])
MRNSample_Visits1 = MRNSample_Visits1 [['Arrival','Departure']]


# Check if Next_Arrival is greater than Departure+2Days
MRNSample_Visits1['Next_Arrival_Dep_Gre2'] = np.where((MRNSample_Visits1['Arrival'].shift(-1))>((MRNSample_Visits1['Departure']) + pd.Timedelta(days=2)),1,0)
