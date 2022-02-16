import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from datetime import date, time, datetime
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn


############################################ Data Cleaning Start (A) ###################################################

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

# Creating the wait times
excelDataFrame["WaitingTime"] = (excelDataFrame["AppointmentDay"]-excelDataFrame["ScheduledDay"]).dt.total_seconds()/(60*60*24)
# Checking for appointment that don't make sense (where the appointment happens before the scheduled date)
print("Number of appointments before their schedule time (Negative): "+str((excelDataFrame["WaitingTime"] < 0).sum()))
# Turning them to positive
for index, row in excelDataFrame.iterrows():
    if row["WaitingTime"] < 0:
        excelDataFrame.loc[index,'WaitingTime'] = row["WaitingTime"] * -1
print("Number of appointments before their schedule time (Negative) after fix: "+str((excelDataFrame["WaitingTime"] < 0).sum()))

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
# pd.set_option("display.max_rows", None, "display.max_columns", None)
# excelDataFrame.to_excel("output.xlsx")

correlationDataframe = excelDataFrame.filter(['SMS_received', 'NormalizedAge', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap','NeighbourhoodInt','NoShowInt', 'GenderInt', 'WaitingTime', \
                            'ScheduleHour', 'ScheduleMonth','ScheduleDayofWeek',''], axis=1)
correlationDataframe["NormalizedAge"] = pd.to_numeric(correlationDataframe["NormalizedAge"])
correlationDataframe["Scholarship"] = pd.to_numeric(correlationDataframe["Scholarship"])
correlationDataframe["Hipertension"] = pd.to_numeric(correlationDataframe["Hipertension"])
correlationDataframe["Diabetes"] = pd.to_numeric(correlationDataframe["Diabetes"])
correlationDataframe["Alcoholism"] = pd.to_numeric(correlationDataframe["Alcoholism"])
correlationDataframe["Handcap"] = pd.to_numeric(correlationDataframe["Handcap"])
correlationDataframe["NeighbourhoodInt"] = pd.to_numeric(correlationDataframe["NeighbourhoodInt"])
correlationDataframe["NoShowInt"] = pd.to_numeric(correlationDataframe["NoShowInt"])
correlationDataframe["GenderInt"] = pd.to_numeric(correlationDataframe["GenderInt"])
correlationDataframe["WaitingTime"] = pd.to_numeric(correlationDataframe["WaitingTime"])
correlationDataframe["ScheduleHour"] = pd.to_numeric(correlationDataframe["ScheduleHour"])
correlationDataframe["ScheduleMonth"] = pd.to_numeric(correlationDataframe["ScheduleMonth"])
correlationDataframe["ScheduleDayofWeek"] = pd.to_numeric(correlationDataframe["ScheduleDayofWeek"])

corrMatrix = correlationDataframe.corr()
sn.heatmap(corrMatrix, annot=True)
# plt.show()

############################################## Data Cleaning End (A) ###################################################

######################################## Model Development Start (B) ###################################################

# Creating x (features) and y (label) series
xDataFrame = correlationDataframe.filter(['SMS_received', 'NormalizedAge', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap','NeighbourhoodInt', 'GenderInt', 'WaitingTime', \
                            'ScheduleHour', 'ScheduleMonth','ScheduleDayofWeek'], axis=1).squeeze()
yDataFrame = correlationDataframe.filter(['NoShowInt'], axis=1).squeeze()

# Splitting the data with a 30% split of testing data
from sklearn.model_selection import train_test_split
XTrainSet, XTestSet, yTrainSet, yTestSet = train_test_split(xDataFrame,yDataFrame, test_size=0.3,random_state=0)

# Creating Naive Bayes classifier pipeline
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
multinomialNBTextClf = Pipeline([('clf', MultinomialNB()), ])

# Getting cross validation score information
from sklearn.model_selection import cross_val_score
multinomialNBTextClfScore = cross_val_score(multinomialNBTextClf, XTrainSet, yTrainSet, cv=5)
print("The 10 fold cross validation score for MultinomialNB is : "+str(multinomialNBTextClfScore))
print("MultinomialNB: %0.2f accuracy with a standard deviation of %0.2f" % (multinomialNBTextClfScore.mean(), multinomialNBTextClfScore.std()))

# Training our model and predicting scores
multinomialNBTextClf.fit(XTrainSet, yTrainSet)
multinomialNBTextClfPredicted = multinomialNBTextClf.predict(XTestSet)

from sklearn import metrics
print("MultinomialNB: Classification report:")
print(metrics.classification_report(yTestSet, multinomialNBTextClfPredicted))

# Getting number of correct predictions and percentage
correct = 0
total = 0
for result in range(len(yTestSet.tolist())):
    total = total + 1
    if yTestSet.tolist()[result] == multinomialNBTextClfPredicted.tolist()[result]:
        correct = correct + 1
print(str(correct)+" correct predictions out of "+str(total))
print("The percentage of the correct predictions is: "+str(correct/total))