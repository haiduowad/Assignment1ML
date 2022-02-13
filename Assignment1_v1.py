import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, time, datetime
from sklearn.preprocessing import MinMaxScaler

# Creating class the will describe the patient information
class PatientInformation():
    def __init__(self, id, appointmentID, gender, age, scheduleDay, scheduleTime, neighbourhood, scholarship, hypertension, diabetes, alcoholism, \
                 handicap, smsReceived, noShow):
        self.id             = id
        self.appointmentID  = appointmentID
        self.gender         = gender
        self.age            = age
        editedDay = scheduleDay.split("T")[0].split('-')
        editedTime = scheduleTime.split("T")[1].replace('Z', '').split(':')
        self.scheduleDay    = date(int(editedDay[0]), int(editedDay[1]), int(editedDay[2]))
        self.scheduleTime   = time(int(editedTime[0]),int(editedTime[1]),int(editedTime[2]))
        self.neighbourhood  = neighbourhood
        self.scholarship    = scholarship
        self.hypertension   = hypertension
        self.diabetes       = diabetes
        self.alcoholism     = alcoholism
        self.handicap       = handicap
        self.smsReceived    = smsReceived
        self.noShow         = noShow

    def checkForMissingData(self):
        if type(self.id) != str:
            if np.isnan(self.id):
                print('Patient number ' + str(self.id) + ' has a missing ID')
        if type(self.appointmentID) != str:
            if np.isnan(self.appointmentID):
                print('Patient number ' + str(self.id) + ' has a missing appointmentID')
        if type(self.gender) != str:
            if np.isnan(self.gender):
                print('Patient number ' + str(self.id) + ' has a missing gender')
        if type(self.age) != str:
            if np.isnan(self.age):
                print('Patient number ' + str(self.id) + ' has a missing age')
        if not isinstance(self.scheduleDay, date):
            if np.isnan(self.scheduleDay):
                print('Patient number ' + str(self.id) + ' has a missing scheduleDay')
        if not isinstance(self.scheduleTime, time):
            if np.isnan(self.scheduleTime):
                print('Patient number ' + str(self.id) + ' has a missing scheduleTime')
        if type(self.neighbourhood) != str:
            if np.isnan(self.neighbourhood):
                print('Patient number ' + str(self.id) + ' has a missing neighbourhood')
        if type(self.scholarship) != str:
            if np.isnan(self.scholarship):
                print('Patient number ' + str(self.id) + ' has a missing scholarship')
        if type(self.hypertension) != str:
            if np.isnan(self.hypertension):
                print('Patient number ' + str(self.id) + ' has a missing hypertension')
        if type(self.diabetes) != str:
            if np.isnan(self.diabetes):
                print('Patient number ' + str(self.id) + ' has a missing diabetes')
        if type(self.alcoholism) != str:
            if np.isnan(self.alcoholism):
                print('Patient number ' + str(self.id) + ' has a missing alcoholism')
        if type(self.handicap) != str:
            if np.isnan(self.handicap):
                print('Patient number ' + str(self.id) + ' has a missing handicap')
        if type(self.smsReceived) != str:
            if np.isnan(self.smsReceived):
                print('Patient number ' + str(self.id) + ' has a missing smsReceived')
        if type(self.noShow) != str:
            if np.isnan(self.noShow):
                print('Patient number ' + str(self.id) + ' has a missing noShow')

    def checkForNegativeAge(self):
        if self.age < 0:
            print('Patient with ID '+str(self.id)+' has a negative age')
            return True

# Importing the patent information
excelDataFrame = pd.read_excel (r'MedicalCentre.xlsx')
#excelDataFrame = pd.read_excel (r'MedicalCentre2.xlsx')

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

# Function that plots a feautre
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
            print("Patient ID "+str(row['PatientId'])+" has a negative age. Removing it from our list.")
            dataFrame.drop(index)
    return dataFrame

excelDataFrame = negativeAgeCleaner(excelDataFrame)

neighbourhoodToIntDict = {}
for neighbourhood in range(len(uniqueNeighbourhood)):
    tempDict = {uniqueNeighbourhood[neighbourhood]:neighbourhood}
    neighbourhoodToIntDict.update(tempDict)

# Function the replaces neighbourhoods with their equivalent int
def neighbourhoodToInt(dataFrame):
    for key in neighbourhoodToIntDict:
        dataFrame['Neighbourhood'] = dataFrame['Neighbourhood'].replace([key], neighbourhoodToIntDict[key])
    return dataFrame

excelDataFrame = neighbourhoodToInt(excelDataFrame)

# Normalizing age
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


# Function the replaces age with their equivalent normalized age
def AgeToNormalizedAge(dataFrame):
    for key in AgeToNormalizedAgeDict:
        dataFrame['Age'] = dataFrame['Age'].replace([key], AgeToNormalizedAgeDict[key])
    return dataFrame

excelDataFrame = AgeToNormalizedAge(excelDataFrame)

# Creating a list of patient instances to store all patient details
patientList = []
for index, row in excelDataFrame.iterrows():
    patientList.append(PatientInformation(row['PatientId'], row['AppointmentID'], row['Gender'], row['Age'], row['AppointmentDay'], row['ScheduledDay'], \
                                          row['Neighbourhood'], row['Scholarship'], row['Hipertension'], row['Diabetes'], row['Alcoholism'], \
                                          row['Handcap'], row['SMS_received'], row['No-show']))

# Checking for mistakes in data
for patient in patientList:
    patient.checkForMissingData()
    patient.checkForNegativeAge()