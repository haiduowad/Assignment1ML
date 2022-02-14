import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, time, datetime
from sklearn.preprocessing import MinMaxScaler

# Creating class the will describe the patient information
class PatientInformation():
    def __init__(self, id, appointmentID, gender, age, scheduleDay, scheduleTime, neighbourhood, year, month, day, hour, minute, second, scholarship, hypertension, diabetes, alcoholism, \
                 handicap, smsReceived, noShow, normalizedAge, neighbourhoodInt):
        self.id             = id
        self.appointmentID  = appointmentID
        self.gender         = gender
        self.age            = age
        self.scheduleDay    = scheduleDay
        self.scheduleTime   = scheduleTime
        self.year           = year
        self.month          = month
        self.day            = day
        self.hour           = hour
        self.minute         = minute
        self.second         = second
        self.neighbourhood  = neighbourhood
        self.scholarship    = scholarship
        self.hypertension   = hypertension
        self.diabetes       = diabetes
        self.alcoholism     = alcoholism
        self.handicap       = handicap
        self.smsReceived    = smsReceived
        self.noShow         = noShow
        self.normalizedAge  = normalizedAge
        self.neighbourhoodInt   = neighbourhoodInt

# Importing the patent information
excelDataFrame = pd.read_excel (r'MedicalCentre.xlsx')
#excelDataFrame = pd.read_excel (r'MedicalCentre2.xlsx')

# Check for empty cells and removing them
emptyCells = np.where(pd.isnull(excelDataFrame))
for index in range(len(emptyCells[0])):
    print("Empty cell found at ["+str(emptyCells[0][index])+", "+str(emptyCells[1][index])+"]")
    print("Removing row number "+str(emptyCells[0][index]))
    excelDataFrame = excelDataFrame.drop(emptyCells[0][index])

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

# Function that removes rows with negative ages
def negativeAgeCleaner(dataFrame):
    for index, row in dataFrame.iterrows():
        if row['Age'] < 0:
            print("Row "+str(index)+" has a negative age. Removing it from our list.")
            dataFrame = dataFrame.drop(index)
    return dataFrame

excelDataFrame = negativeAgeCleaner(excelDataFrame)

# Creating dictionary with neighbourhood as integers
neighbourhoodToIntDict = {}
for neighbourhood in range(len(uniqueNeighbourhood)):
    tempDict = {uniqueNeighbourhood[neighbourhood]:neighbourhood}
    neighbourhoodToIntDict.update(tempDict)

# Creating column for integer value of the neighbourhood and populating it
excelDataFrame["NeighbourhoodInt"] = ""
for index, row in excelDataFrame.iterrows():
    excelDataFrame.at[index, 'NeighbourhoodInt'] = neighbourhoodToIntDict[row["Neighbourhood"]]

# Creating columns for date and time and populating them
excelDataFrame["Year"] = ""
excelDataFrame["Month"] = ""
excelDataFrame["Day"] = ""
excelDataFrame["Hour"] = ""
excelDataFrame["Minute"] = ""
excelDataFrame["Second"] = ""
for index, row in excelDataFrame.iterrows():
    editedDay = row["AppointmentDay"].split("T")[0].split('-')
    editedTime = row["ScheduledDay"].split("T")[1].replace('Z', '').split(':')
    excelDataFrame.at[index, 'Year'] = editedDay[0]
    excelDataFrame.at[index, 'Month'] = editedDay[1]
    excelDataFrame.at[index, 'Day'] = editedDay[2]
    excelDataFrame.at[index, 'Hour'] = editedTime[0]
    excelDataFrame.at[index, 'Minute'] = editedTime[1]
    excelDataFrame.at[index, 'Second'] = editedTime[2]

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

# Creating excel sheet for the cleaned data
excelDataFrame.to_excel("output.xlsx")

# Creating a list of patient instances to store all patient details
patientList = []
for index, row in excelDataFrame.iterrows():
    patientList.append(PatientInformation(row['PatientId'], row['AppointmentID'], row['Gender'], row['Age'], row['AppointmentDay'], row['ScheduledDay'], \
                                          row['Neighbourhood'], row['Year'], row['Month'], row['Day'], row['Hour'], row['Minute'], row['Second'], row['Scholarship'], row['Hipertension'], row['Diabetes'], row['Alcoholism'], \
                                          row['Handcap'], row['SMS_received'], row['No-show'], row['NormalizedAge'], row['NeighbourhoodInt']))