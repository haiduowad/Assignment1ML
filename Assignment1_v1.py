import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from datetime import date, time, datetime
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn


################################################ Data Cleaning Start ###################################################

# Importing the patent information
excelDataFrame = pd.read_excel (r'MedicalCentre.xlsx')
# excelDataFrame = pd.read_excel (r'MedicalCentre2.xlsx')

# Check for empty cells and removing them
emptyCells = np.where(pd.isnull(excelDataFrame))
for index in range(len(emptyCells[0])):
    print("Empty cell found at ["+str(emptyCells[0][index])+", "+str(emptyCells[1][index])+"]")
    print("Removing row number "+str(emptyCells[0][index]))
    excelDataFrame = excelDataFrame.drop(emptyCells[0][index])

# Checking for duplicate appointments and removing them
duplicateAppointments = excelDataFrame['AppointmentID'].duplicated().tolist()
for index in range(len(duplicateAppointments)):
    if duplicateAppointments[index] == True:
        print("There is a duplicate appointment at index "+str(index))
        print("Dropping index "+str(index))
        excelDataFrame = excelDataFrame.drop(index)

# Getting number of unique values in a feature set
uniquePatientId = pd.unique(excelDataFrame[["PatientId"]].values.ravel())
uniqueAppointmentId = pd.unique(excelDataFrame[["AppointmentID"]].values.ravel())
uniqueGender = pd.unique(excelDataFrame[["Gender"]].values.ravel())
uniqueTime = pd.unique(excelDataFrame[["ScheduledDay"]].values.ravel())
uniqueDay = pd.unique(excelDataFrame[["AppointmentDay"]].values.ravel())
uniqueAge = pd.unique(excelDataFrame[["Age"]].values.ravel())
uniqueNeighbourhood = pd.unique(excelDataFrame[["Neighbourhood"]].values.ravel())
uniqueScholarship = pd.unique(excelDataFrame[["Scholarship"]].values.ravel())
uniqueHypertension = pd.unique(excelDataFrame[["Hipertension"]].values.ravel())
uniqueDiabetes = pd.unique(excelDataFrame[["Diabetes"]].values.ravel())
uniqueAlcoholism = pd.unique(excelDataFrame[["Alcoholism"]].values.ravel())
uniqueHandicap = pd.unique(excelDataFrame[["Handcap"]].values.ravel())
uniqueSmsReceived = pd.unique(excelDataFrame[["SMS_received"]].values.ravel())
uniqueNoShow = pd.unique(excelDataFrame[["No-show"]].values.ravel())

print('Unique number of Patient IDs: '+str(len(uniquePatientId)))
print('Unique number of Appointment IDs: '+str(len(uniqueAppointmentId)))
print('Unique number of Genders: '+str(len(uniqueGender)))
print('Unique number of Appointment Times: '+str(len(uniqueTime)))
print('Unique number of Appointment Days: '+str(len(uniqueDay)))
print('Unique number of Ages: '+str(len(uniqueAge)))
print('Unique number of Neighbourhoods: '+str(len(uniqueNeighbourhood)))
print('Unique number of Scholarship: '+str(len(uniqueScholarship)))
print('Unique number of Hypertension: '+str(len(uniqueHypertension)))
print('Unique number of Diabetes: '+str(len(uniqueDiabetes)))
print('Unique number of Alcoholism: '+str(len(uniqueAlcoholism)))
print('Unique number of Handicap: '+str(len(uniqueHandicap)))
print('Unique number of SMS Received: '+str(len(uniqueSmsReceived)))
print('Unique number of No Shows: '+str(len(uniqueNoShow)))

# Function that plots a feature
def featurePlotter(xValues, yValues, yName):
    myList = []
    xList = []
    yList = []
    for xValue in range(len(xValues)):
        xList.append(str(xValues[xValue]))
        yList.append(yValues[xValue])
    myList.append(xList)
    myList.append(yList)
    plt.scatter(myList[:][0], myList[:][1])
    plt.suptitle('Patient ID vs '+yName, fontsize=20)
    plt.xlabel('Patient ID', fontsize=18)
    plt.ylabel(yName, fontsize=16)
    plt.show()

#featurePlotter(excelDataFrame["PatientId"], excelDataFrame["Age"], "Age")
#featurePlotter(excelDataFrame["PatientId"], excelDataFrame["ScheduledDay"], "Appointment Times")

# Checking for outliers (for age)
print("The highest age is: "+ str(excelDataFrame["Age"].max()))
print("The lowest age is: "+ str(excelDataFrame["Age"].min()))

# Removing negative ages
for index, row in excelDataFrame.iterrows():
    if row['Age'] < 0:
        print("Row "+str(index)+" has a negative age. Removing it from our list.")
        excelDataFrame = excelDataFrame.drop(index)

# Changing handicap values to 1 if greater
for index, row in excelDataFrame.iterrows():
    if row['Handcap'] > 0:
        excelDataFrame.at[index, 'Handcap'] = int(1)

# Creating dictionary with neighbourhood as integers
neighbourhoodToIntDict = {}
for neighbourhood in range(len(uniqueNeighbourhood)):
    tempDict = {uniqueNeighbourhood[neighbourhood]:neighbourhood}
    neighbourhoodToIntDict.update(tempDict)

# Creating column for integer value of the neighbourhood and populating it
excelDataFrame["NeighbourhoodInt"] = ""
for index, row in excelDataFrame.iterrows():
    excelDataFrame.at[index, 'NeighbourhoodInt'] = neighbourhoodToIntDict[row["Neighbourhood"]]

# Changing dates to datetime format
excelDataFrame['ScheduledDay'] = pd.to_datetime(excelDataFrame['ScheduledDay'], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
excelDataFrame['AppointmentDay'] = pd.to_datetime(excelDataFrame['AppointmentDay'], format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")

# Creating columns for date and time and populating them
excelDataFrame["AppointmentYear"] = excelDataFrame["AppointmentDay"].dt.year
excelDataFrame["AppointmentMonth"] = excelDataFrame["AppointmentDay"].dt.month
excelDataFrame["AppointmentDay1"] = excelDataFrame["AppointmentDay"].dt.day
excelDataFrame["AppointmentHour"] = excelDataFrame["AppointmentDay"].dt.hour
excelDataFrame["AppointmentMinute"] = excelDataFrame["AppointmentDay"].dt.minute
excelDataFrame["AppointmentSecond"] = excelDataFrame["AppointmentDay"].dt.second
excelDataFrame["AppointmentDayofWeek"] = excelDataFrame["AppointmentDay"].dt.dayofweek
excelDataFrame["ScheduleYear"] = excelDataFrame["ScheduledDay"].dt.year
excelDataFrame["ScheduleMonth"] = excelDataFrame["ScheduledDay"].dt.month
excelDataFrame["ScheduleDay"] = excelDataFrame["ScheduledDay"].dt.day
excelDataFrame["ScheduleHour"] = excelDataFrame["ScheduledDay"].dt.hour
excelDataFrame["ScheduleMinute"] = excelDataFrame["ScheduledDay"].dt.minute
excelDataFrame["ScheduleSecond"] = excelDataFrame["ScheduledDay"].dt.second
excelDataFrame["ScheduleDayofWeek"] = excelDataFrame["ScheduledDay"].dt.dayofweek

# Checking for appointment that don't make sense (where the appointment happens before the scheduled date)
print("Number of appointments before their schedule time: "+str((excelDataFrame["ScheduledDay"] > excelDataFrame["AppointmentDay"]).sum()))
# Making the appointment day at the end of the day to get time component to fix this issue
excelDataFrame['AppointmentDay'] = excelDataFrame['AppointmentDay'] + pd.Timedelta('1d') - pd.Timedelta('1s')
# Now check for improvements
print("Number of appointments before their schedule time after fix: "+str((excelDataFrame["ScheduledDay"] > excelDataFrame["AppointmentDay"]).sum()))
# Remove the bad rows
print("Removing these rows")
excelDataFrame = excelDataFrame.loc[(excelDataFrame["AppointmentDay"] >= excelDataFrame["ScheduledDay"])].copy()

# Creating the wait times
excelDataFrame["WaitingTime"] = (excelDataFrame["AppointmentDay"]-excelDataFrame["ScheduledDay"]).dt.total_seconds()/(60*60*24)

# Creating dictionary with normalized ages
AgeMinMax = MinMaxScaler()
Ages = []
for age in uniqueAge:
    Ages.append([age])
AgeMinMax.fit(Ages)
AgesScaled = AgeMinMax.transform(Ages)
AgeToNormalizedAgeDict = {}
for age in range(len(Ages)):
    tempDict = {Ages[age][0]: AgesScaled[age][0]}
    AgeToNormalizedAgeDict.update(tempDict)

# Creating column for integer value of the neighbourhood and populating it
excelDataFrame["NormalizedAge"] = ""
for index, row in excelDataFrame.iterrows():
    excelDataFrame.at[index, 'NormalizedAge'] = AgeToNormalizedAgeDict[row["Age"]]

# Changing no-show to int values (no-show == 1)
excelDataFrame["NoShowInt"] = ""
for index, row in excelDataFrame.iterrows():
    if row["No-show"] == "Yes":
        excelDataFrame.at[index, 'NoShowInt'] = int(1)
    elif row["No-show"] == "No":
        excelDataFrame.at[index, 'NoShowInt'] = int(0)

# Changing gender to int (M == 1)
excelDataFrame["GenderInt"] = ""
for index, row in excelDataFrame.iterrows():
    if row["Gender"] == "M":
        excelDataFrame.at[index, 'GenderInt'] = int(1)
    elif row["Gender"] == "F":
        excelDataFrame.at[index, 'GenderInt'] = int(0)

# Creating excel sheet for the cleaned data
# excelDataFrame.to_excel("output.xlsx")

df = excelDataFrame.filter(['SMS_received', 'NormalizedAge', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap','NeighbourhoodInt','NoShowInt', 'GenderInt'], axis=1)
df["NormalizedAge"] = pd.to_numeric(df["NormalizedAge"])
df["Scholarship"] = pd.to_numeric(df["Scholarship"])
df["Hipertension"] = pd.to_numeric(df["Hipertension"])
df["Diabetes"] = pd.to_numeric(df["Diabetes"])
df["Alcoholism"] = pd.to_numeric(df["Alcoholism"])
df["Handcap"] = pd.to_numeric(df["Handcap"])
df["NeighbourhoodInt"] = pd.to_numeric(df["NeighbourhoodInt"])
df["NoShowInt"] = pd.to_numeric(df["NoShowInt"])
df["GenderInt"] = pd.to_numeric(df["GenderInt"])
#pd.set_option("display.max_rows", None, "display.max_columns", None)
corrMatrix = df.corr()
print(corrMatrix)
sn.heatmap(corrMatrix, annot=True)
plt.show()

################################################## Data Cleaning End ###################################################

#finalData = excelDataFrame.filter([''], axis=1)