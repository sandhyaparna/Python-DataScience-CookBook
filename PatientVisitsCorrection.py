# Data
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
