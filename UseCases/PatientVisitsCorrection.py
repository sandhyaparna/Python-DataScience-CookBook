# Problem Statement
# A Patient is given different visit IDs every time they visit a hospital. 
# Some times even during the same visit of a patient they give different patient IDs.
# So based on Arrival and Departure dates we should identify 'Actual Arrival' and 'Actual Departure' for each visit of a patient
# Departure should be after or at Arrival time 
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
# Within each MRN - Next_Arrival_Dep_Gre2 should be 1 except the last observation

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

#### Iterate the next steps

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

MRNSample_Visits1

# Within a MRN if Next_Arrival_Dep_Gre2 is not 1 for all but the last obs, run the code from the line 'Iterate the next steps'

----------------
####### Multiple MRNs ########
# Load Data Frames
import pickle
VigiID = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/VigiID.pkl")
AllVisits_DM = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/AllVisits_DM.pkl")
Patient_Stay = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Patient_Stay.pkl")
Freq_MRNs_1 = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Freq_MRNs_1.pkl")
Freq_MRNs_2 = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Freq_MRNs_2.pkl")
Patient_Visits_Freq1 = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Patient_Visits_Freq1.pkl")
Patient_Visits = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Patient_Visits.pkl")
Patient_Visits_Corr = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Patient_Visits_Corr.pkl")
 = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/.pkl")
 = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/.pkl")
 = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/.pkl")
 = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/.pkl")
 = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/.pkl")
 = pd.read_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/.pkl")


### Analysis is based on Admission and Discharge ###
# Extract required columns
Patient_Visits = Patient_Stay[['VigiClientID','UniqueID','AdmissionDateTime','DischargeDateTime']]

# Join MRN to Patient_Visits
Patient_Visits = Patient_Visits.merge(VigiID, how='inner', on=['VigiClientID','UniqueID'])

# Check for cases where Admission is before Discharge
AdmBfrDis = Patient_Visits[Patient_Visits.DischargeDateTime<Patient_Visits.AdmissionDateTime]
# Delete cases where Discharge is before Admission
Patient_Visits = Patient_Visits[Patient_Visits.DischargeDateTime>=Patient_Visits.AdmissionDateTime]

# There are no cases where AdmissionDateTime is after DischargeDateTime - There are a very few cases where AdmissionDateTime=DischargeDateTime 

# Freq of MRNs visiting once, twice etc
Freq_MRNs = Patient_Visits[['VigiClientID','MRN']]
Freq_MRNs = Freq_MRNs.assign(ID=1)
Freq_MRNs = pd.DataFrame(Freq_MRNs.groupby(['VigiClientID','MRN']).size()).reset_index()
Freq_MRNs = Freq_MRNs.rename(columns={0:"Freq"})
# Patients/MRNs that have 1 visit only - their Visit IDs cannot be combined, so seperate them
Freq_MRNs_1 = Freq_MRNs[Freq_MRNs.Freq==1]
# Save as py dataframe
Freq_MRNs_1.to_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Freq_MRNs_1.pkl")
# Patients/MRNs that have atleast 2 visits
Freq_MRNs_2 = Freq_MRNs[Freq_MRNs.Freq>=2]
# Save as py dataframe
Freq_MRNs_2.to_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Freq_MRNs_2.pkl")

# Extract visits of Patients that visited just once
Patient_Visits_Freq1 = Patient_Visits[Patient_Visits.MRN.isin(Freq_MRNs_1.MRN)]

# Extract all visits of Patients that visited more than once
Patient_Visits = Patient_Visits[Patient_Visits.MRN.isin(Freq_MRNs_2.MRN)]

# save
Patient_Visits_Freq1.to_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Patient_Visits_Freq1.pkl")
Patient_Visits.to_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Patient_Visits.pkl")

### Logic for combining visits ###
# Extract & Sort data by  MRN,AdmissionDateTime,DischargeDateTime
Patient_Visits_Corr = Patient_Visits[['MRN','AdmissionDateTime','DischargeDateTime']]
Patient_Visits_Corr = Patient_Visits_Corr.sort_values(by=['MRN','AdmissionDateTime','DischargeDateTime'])

# Remove duplicate AdmissionDateTimes - within each AdmissionDateTime pick max DischargeDateTime
import pip
from pandasql import *
import pandasql as ps
query = """select MRN,AdmissionDateTime,max(DischargeDateTime) as DischargeDateTime from Patient_Visits_Corr group by MRN,AdmissionDateTime"""
Patient_Visits_Corr = ps.sqldf(query, locals())

# Convert AdmissionDateTime,DischargeDateTime to datetime
import datetime
Patient_Visits_Corr['AdmissionDateTime'] =  pd.to_datetime(Patient_Visits_Corr['AdmissionDateTime'], format='%Y-%m-%d %H:%M:%S.%f')
Patient_Visits_Corr['DischargeDateTime'] =  pd.to_datetime(Patient_Visits_Corr['DischargeDateTime'], format='%Y-%m-%d %H:%M:%S.%f')

# Check if Next_AdmissionDateTime is greater than DischargeDateTime+2Days
Patient_Visits_Corr['Next_Adm_Dis_Gre1'] = np.where((Patient_Visits_Corr.groupby(['MRN'])['AdmissionDateTime'].shift(-1))>((Patient_Visits_Corr['DischargeDateTime']) + pd.Timedelta(days=1)),1,0)

# Check if all values of Next_Adm_Dis_Gre1 are 1 except last obs within a MRN
# Calculate number of times each MRN occurs - Compare it with sum(Next_Adm_Dis_Gre1) - Difference should be exactly 1 for each MRN - otherwise run the code again
Next_Adm_Dis_Gre1 = Patient_Visits_Corr.assign(ID=1)
Next_Adm_Dis_Gre1 = pd.DataFrame(Next_Adm_Dis_Gre1.groupby('MRN').agg({'ID':'sum','Next_Adm_Dis_Gre1':'sum'})).reset_index()
Next_Adm_Dis_Gre1['Diff'] = Next_Adm_Dis_Gre1['ID']-Next_Adm_Dis_Gre1['Next_Adm_Dis_Gre1']
# filter to see if any MRN has Diff>1
len(Next_Adm_Dis_Gre1[Next_Adm_Dis_Gre1.Diff>1])
# If len(Next_Adm_Dis_Gre1[Next_Adm_Dis_Gre1.Diff>1])>0 run the below code


#### 1st Iteration,2nd,3

# Self join data using MRN
Patient_Visits_Corr = Patient_Visits_Corr[['MRN','AdmissionDateTime','DischargeDateTime']].merge(Patient_Visits_Corr[['MRN','AdmissionDateTime','DischargeDateTime']], how='inner',on='MRN')

# Since there are no duplicate AdmissionDateTimes
# Delete obs where AdmissionDateTime_y<AdmissionDateTime_x and not AdmissionDateTime_y<=AdmissionDateTime_x because id last patientID is unique to itself - it goes away
# Patient_Visits_Corr = Patient_Visits_Corr[(Patient_Visits_Corr.AdmissionDateTime_y>=Patient_Visits_Corr.AdmissionDateTime_x)]

# Check if time difference between A_y & A_x is >=0 & A_y &D_x<=2days if yes then Ay=Ax and Dy is max(Dx,Dy)
Patient_Visits_Corr['DischargeDateTime_New'] = np.where(( ((Patient_Visits_Corr['AdmissionDateTime_y'])<=((Patient_Visits_Corr['DischargeDateTime_x'])+(pd.Timedelta(days=1)))) & ((Patient_Visits_Corr.AdmissionDateTime_y)>=(Patient_Visits_Corr.AdmissionDateTime_x)) ), 
                                                np.maximum(Patient_Visits_Corr['DischargeDateTime_x'],Patient_Visits_Corr['DischargeDateTime_y']),
                                                Patient_Visits_Corr['DischargeDateTime_x'])

Patient_Visits_Corr['DischargeDateTime_New1'] = np.where(( ((Patient_Visits_Corr['AdmissionDateTime_x'])<=((Patient_Visits_Corr['DischargeDateTime_y'])+(pd.Timedelta(days=1)))) & ((Patient_Visits_Corr.AdmissionDateTime_x)>=(Patient_Visits_Corr.AdmissionDateTime_y)) ), 
                                                np.maximum(Patient_Visits_Corr['DischargeDateTime_x'],Patient_Visits_Corr['DischargeDateTime_y']),
                                                Patient_Visits_Corr['DischargeDateTime_x'])

# Extract AdmissionDateTime_x and DischargeDateTime_New
Patient_Visits_Corra = Patient_Visits_Corr[['MRN','AdmissionDateTime_x','DischargeDateTime_New']].drop_duplicates()
Patient_Visits_Corra = Patient_Visits_Corra.rename(columns={"DischargeDateTime_New":"DischargeDateTime"})

Patient_Visits_Corrb = Patient_Visits_Corr[['MRN','AdmissionDateTime_x','DischargeDateTime_New1']].drop_duplicates()
Patient_Visits_Corrb = Patient_Visits_Corrb.rename(columns={"DischargeDateTime_New1":"DischargeDateTime"})

Patient_Visits_Corr = pd.concat([Patient_Visits_Corra,Patient_Visits_Corrb])

# Remove duplicate AdmissionDateTimes - within each AdmissionDateTime pick max DischargeDateTime
import pip
from pandasql import *
import pandasql as ps
query = """select MRN,AdmissionDateTime_x as AdmissionDateTime,max(DischargeDateTime) as DischargeDateTime from Patient_Visits_Corr group by MRN,AdmissionDateTime_x"""
Patient_Visits_Corr = ps.sqldf(query, locals())

# Convert AdmissionDateTime,DischargeDateTime to datetime
import datetime
Patient_Visits_Corr['AdmissionDateTime'] =  pd.to_datetime(Patient_Visits_Corr['AdmissionDateTime'], format='%Y-%m-%d %H:%M:%S.%f')
Patient_Visits_Corr['DischargeDateTime'] =  pd.to_datetime(Patient_Visits_Corr['DischargeDateTime'], format='%Y-%m-%d %H:%M:%S.%f')

# Remove duplicate AdmissionDateTimes - within each DischargeDateTime pick min AdmissionDateTime
import pip
from pandasql import *
import pandasql as ps
query = """select MRN,DischargeDateTime,min(AdmissionDateTime) as AdmissionDateTime from Patient_Visits_Corr group by MRN,DischargeDateTime"""
Patient_Visits_Corr = ps.sqldf(query, locals()) 

# Convert AdmissionDateTime,DischargeDateTime to datetime
import datetime
Patient_Visits_Corr['AdmissionDateTime'] =  pd.to_datetime(Patient_Visits_Corr['AdmissionDateTime'], format='%Y-%m-%d %H:%M:%S.%f')
Patient_Visits_Corr['DischargeDateTime'] =  pd.to_datetime(Patient_Visits_Corr['DischargeDateTime'], format='%Y-%m-%d %H:%M:%S.%f')

# Sort data by AdmissionDateTime
Patient_Visits_Corr = Patient_Visits_Corr.sort_values(by=['MRN','AdmissionDateTime'])

# Check if Next_AdmissionDateTime is greater than DischargeDateTime+1Days
Patient_Visits_Corr['Next_Adm_Dis_Gre1'] = np.where((Patient_Visits_Corr.groupby(['MRN'])['AdmissionDateTime'].shift(-1))>((Patient_Visits_Corr['DischargeDateTime']) + pd.Timedelta(days=1)),1,0)

# Check if all values of Next_Adm_Dis_Gre1 are 1 except last obs within a MRN
# Calculate number of times each MRN occurs - Compare it with sum(Next_Adm_Dis_Gre1) - Difference should be exactly 1 for each MRN - otherwise run the code again
Next_Adm_Dis_Gre1 = Patient_Visits_Corr.assign(ID=1)
Next_Adm_Dis_Gre1 = pd.DataFrame(Next_Adm_Dis_Gre1.groupby('MRN').agg({'ID':'sum','Next_Adm_Dis_Gre1':'sum'})).reset_index()
Next_Adm_Dis_Gre1['Diff'] = Next_Adm_Dis_Gre1['ID']-Next_Adm_Dis_Gre1['Next_Adm_Dis_Gre1']
# filter to see if any MRN has Diff>1
Next_Adm_Dis_Gre1 = Next_Adm_Dis_Gre1[Next_Adm_Dis_Gre1.Diff>1]
len(Next_Adm_Dis_Gre1[Next_Adm_Dis_Gre1.Diff>1])
# If len(Next_Adm_Dis_Gre1[Next_Adm_Dis_Gre1.Diff>1])>0 run the iteration part
       
####### Iteration part over ########
Patient_Visits_Corr = Patient_Visits_Corr[['MRN','AdmissionDateTime','DischargeDateTime']]
# save as py data frame
Patient_Visits_Corr.to_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Patient_Visits_Corr.pkl")

# Interesting cases
Iter1 = Patient_Visits[Patient_Visits.MRN.isin(pd.unique(Next_Adm_Dis_Gre1['MRN']))]
Iter1.to_csv('C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Iter1.csv',index=False)

Iter2 = Patient_Visits[Patient_Visits.MRN.isin(pd.unique(Next_Adm_Dis_Gre1['MRN']))]
Iter2.to_csv('C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Iter2.csv',index=False)

Iter3 = Patient_Visits[Patient_Visits.MRN.isin(pd.unique(Next_Adm_Dis_Gre1['MRN']))]
Iter3.to_csv('C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Iter3.csv',index=False)

Iter4 = Patient_Visits[Patient_Visits.MRN.isin(pd.unique(Next_Adm_Dis_Gre1['MRN']))]
Iter4.to_csv('C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Iter4.csv',index=False)

Iter5 = Patient_Visits[Patient_Visits.MRN.isin(pd.unique(Next_Adm_Dis_Gre1['MRN']))]
Iter5.to_csv('C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Iter5.csv',index=False)


##### Join back the correct Admission discharges to MRN-UniqueID ######
# Corresponds to patients that have just 1 visitID
Patient_Visits_Freq1

# Corresponds to patients that have atleast 2 Visit IDs
Patient_Visits

# Join above 2 datasets
Patient_Stay_Corr = pd.concat([Patient_Visits,Patient_Visits_Freq1])

# Rename columns of Patient_Visits_Corr 
Patient_Visits_Freq2_Corr = Patient_Visits_Corr.rename(columns={"AdmissionDateTime":"New_AdmissionDateTime","DischargeDateTime":"New_DischargeDateTime"})
Patient_Visits_Freq2_Corr = Patient_Visits_Freq2_Corr.assign(AdmissionDateTime=Patient_Visits_Freq2_Corr.New_AdmissionDateTime)

# Extract only required columns from Patient_Visits_Freq1 (There is no correction required as these patients have only 1 visit)
Patient_Visits_Freq1_Corr = Patient_Visits_Freq1[['MRN','AdmissionDateTime','DischargeDateTime']]
Patient_Visits_Freq1_Corr = Patient_Visits_Freq1_Corr.rename(columns={"AdmissionDateTime":"New_AdmissionDateTime","DischargeDateTime":"New_DischargeDateTime"})
Patient_Visits_Freq1_Corr = Patient_Visits_Freq1_Corr.assign(AdmissionDateTime=Patient_Visits_Freq1_Corr.New_AdmissionDateTime)

# Join above 2 data frames
Patient_Visits_All_Corr = pd.concat([Patient_Visits_Freq2_Corr,Patient_Visits_Freq1_Corr])

# Join Patient_Stay_Corr & Patient_Stay_Corr based on (MRN and AdmissionDateTime)
Patient_Stay_Corr = Patient_Stay_Corr.merge(Patient_Visits_All_Corr, how='left', on=['MRN','AdmissionDateTime'])
# Front fill Nas
Patient_Stay_Corr = Patient_Stay_Corr.fillna(method='pad')

# save as py data frame - Update single visits data
Patient_Stay_Corr.to_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Patient_Stay_Corr.pkl")


#### Extract specific MRNs ####
Iter_1 =  pd.read_csv("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Iter1.csv",
                 dtype = {'MRN': str}) # Change data type of a variable) # To import variable in datetime format default format is '%d%b%Y:%H:%M:%S.%f'

Patient_Stay_Corr_Sample = Patient_Stay_Corr[Patient_Stay_Corr.MRN.isin(pd.unique(Iter_1['MRN']))]
Patient_Stay_Corr_Sample.to_pickle("C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Patient_Stay_Corr_Sample.pkl")
Patient_Stay_Corr_Sample.to_csv('C:/Users/spashikanti/Desktop/Readmissions - 2018/Py DataFrames/Patient_Stay_Corr_Sample.csv',index=False)

# Admission Type
Admission_Type = AllVisits_DM[[ 'UniqueID','AdmissionType','PlannedAdmission']]
Patient_Stay_Corr_Sample = Patient_Stay_Corr_Sample.merge(Admission_Type, how='left', on=['UniqueID'])






