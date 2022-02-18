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

################################## A(1) Start

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

################################## A(1) End

################################## A(2) Start

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

################################## A(2) End

################################## A(3) Start

# Function that plots a feature
# Takes in yLabel, yValues (Optional xValues and XLabel)
def featurePlotter(*args):
    myList = []
    yName = args[0]
    yList = args[1]
    if len(args) > 2:
        xList = args[2]
        xName = args[3]
    else:
        xList = []
        for x in range(len(yList)):
            xList.append(x)

    myList.append(xList)
    myList.append(yList)
    plt.scatter(myList[:][0], myList[:][1])

    if len(args) > 2:
        plt.suptitle('Patient ID vs '+yName, fontsize=20)
        plt.xlabel('Patient ID', fontsize=18)
    else:
        plt.suptitle(yName, fontsize=20)
        plt.xlabel('Unique Element', fontsize=18)
    plt.ylabel(yName, fontsize=16)
    plt.show()

featurePlotter("Age", uniqueAge)
featurePlotter("Handicap", uniqueHandicap)
# featurePlotter("Time", uniqueTime)

# Checking for outliers (for age)
print("The highest age is: "+ str(excelDataFrame["Age"].max()))
print("The lowest age is: "+ str(excelDataFrame["Age"].min()))

# Changing handicap values to 1 if greater
for index, row in excelDataFrame.iterrows():
    if row['Handcap'] > 0:
        excelDataFrame.at[index, 'Handcap'] = int(1)

################################## A(3) End

################################## A(4) Start

# Removing negative ages
for index, row in excelDataFrame.iterrows():
    if row['Age'] < 0:
        print("Row "+str(index)+" has a negative age. Removing it from our list.")
        excelDataFrame = excelDataFrame.drop(index)

################################## A(4) End

################################## A(5) & A(7) Start

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

################################## A(5) and A(7) End

################################## A(6) Start

# Creating dictionary with neighbourhood as integers
neighbourhoodToIntDict = {}
for neighbourhood in range(len(uniqueNeighbourhood)):
    tempDict = {uniqueNeighbourhood[neighbourhood]:neighbourhood}
    neighbourhoodToIntDict.update(tempDict)

# Creating column for integer value of the neighbourhood and populating it
excelDataFrame["NeighbourhoodInt"] = ""
for index, row in excelDataFrame.iterrows():
    excelDataFrame.at[index, 'NeighbourhoodInt'] = neighbourhoodToIntDict[row["Neighbourhood"]]

################################## A(6) End

################################## A(8) Start

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

################################## A(8) End

# Creating excel sheet for the cleaned data
# pd.set_option("display.max_rows", None, "display.max_columns", None)
# excelDataFrame.to_excel("output.xlsx")

################################## A(9) Start

# Getting features that were edited to work for our ML algorithm and changing them to numeric
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

# Plotting correlation matrix
corrMatrix = correlationDataframe.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

################################## A(9) End

############################################## Data Cleaning End (A) ###################################################

######################################## Model Development Start (B) ###################################################

# Creating x (features) and y (label) series
# Results1
# xDataFrame = correlationDataframe.filter(['SMS_received', 'NormalizedAge', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap','NeighbourhoodInt', 'GenderInt', 'WaitingTime', \
#                             'ScheduleHour', 'ScheduleMonth','ScheduleDayofWeek'], axis=1).squeeze()
# Results2
# xDataFrame = correlationDataframe.filter(['SMS_received', 'NormalizedAge', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap','NeighbourhoodInt', 'GenderInt', 'WaitingTime'], axis=1).squeeze()
# Results3
# xDataFrame = correlationDataframe.filter(['SMS_received', 'WaitingTime'], axis=1).squeeze()
# Results4
# xDataFrame = correlationDataframe.filter(['SMS_received','NeighbourhoodInt', 'GenderInt', 'WaitingTime'], axis=1).squeeze()
# Results5
# xDataFrame = correlationDataframe.filter(['SMS_received', 'NeighbourhoodInt', 'GenderInt', 'WaitingTime', \
#                              'ScheduleHour', 'ScheduleMonth','ScheduleDayofWeek'], axis=1).squeeze()
# Results6
# xDataFrame = correlationDataframe.filter(['SMS_received', 'NeighbourhoodInt', 'WaitingTime', \
#                              'ScheduleHour','ScheduleDayofWeek'], axis=1).squeeze()
# # Results7
# xDataFrame = correlationDataframe.filter(['SMS_received', 'WaitingTime', 'ScheduleHour','ScheduleDayofWeek'], axis=1).squeeze()
# Results8
xDataFrame = correlationDataframe.filter(['SMS_received', 'WaitingTime', 'ScheduleDayofWeek'], axis=1).squeeze()
yDataFrame = correlationDataframe.filter(['NoShowInt'], axis=1).squeeze()

# Splitting the data with a 30% split of testing data
from sklearn.model_selection import train_test_split
XTrainSet, XTestSet, yTrainSet, yTestSet = train_test_split(xDataFrame,yDataFrame, test_size=0.3,random_state=0)

# Creating Naive Bayes classifier pipeline
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
multinomialNBClf = Pipeline([('clf', MultinomialNB()), ])

# Training our model and predicting scores
multinomialNBClf.fit(XTrainSet, yTrainSet)

######################################### Model Development End (B) ####################################################

######################################### Model Evaluation Start (C) ###################################################

################################## C(1) Start

# Getting 10 fold cross validation score information
from sklearn.model_selection import cross_val_score
multinomialNBClfScore = cross_val_score(multinomialNBClf, XTrainSet, yTrainSet, cv=10)
print("The 10 fold cross validation score for MultinomialNB is : "+str(multinomialNBClfScore))
print("MultinomialNB: %0.2f accuracy with a standard deviation of %0.2f" % (multinomialNBClfScore.mean(), multinomialNBClfScore.std()))

# Getting classification report
from sklearn import metrics
multinomialNBClfPredicted = multinomialNBClf.predict(XTestSet)
print("MultinomialNB: Classification report:")
print(metrics.classification_report(yTestSet.to_numpy().tolist(), multinomialNBClfPredicted.tolist()))

# Getting number of correct predictions and percentage
correct = 0
total = 0
yTestSetNew = yTestSet.tolist()
multinomialNBClfPredictedNew = multinomialNBClfPredicted.tolist()
for result in range(len(yTestSetNew)):
    total = total + 1
    if yTestSetNew[result] == multinomialNBClfPredictedNew[result]:
        correct = correct + 1
print(str(correct)+" correct predictions out of "+str(total)+" for MultinomialNB")
print("The MultinomialNB percentage of the correct predictions is: "+str(correct/total))

################################## C(1) End

################################## C(2) Start

# Getting a range of parameters for multinomial NB classifier
gridParameters = {'clf__alpha': np.linspace(0.5, 1.5, 6), 'clf__fit_prior': [True, False]}
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(multinomialNBClf, gridParameters)
clf.fit(XTrainSet, yTrainSet)

# Getting the best score
print("Best Score: ", clf.best_score_)
print("Best Params: ", clf.best_params_)

################################## C(2) End

################################## C(3) Start

# Setting up Decision Tree and obtaining results
from sklearn.tree import DecisionTreeClassifier
DecisionTreeClf     = Pipeline([('clf', DecisionTreeClassifier()), ])
DecisionTreeClf.fit(XTrainSet, yTrainSet)
DecisionTreeClfScore = cross_val_score(DecisionTreeClf, XTrainSet, yTrainSet, cv=10)
print("The 10 fold cross validation score for DecisionTree is : "+str(DecisionTreeClfScore))
print("DecisionTree: %0.2f accuracy with a standard deviation of %0.2f" % (DecisionTreeClfScore.mean(), DecisionTreeClfScore.std()))
DecisionTreeClfPredicted = DecisionTreeClf.predict(XTestSet)
print("DecisionTree: Classification report:")
print(metrics.classification_report(yTestSet.to_numpy().tolist(), DecisionTreeClfPredicted.tolist()))
correct = 0
total = 0
DecisionTreeClfPredictedNew = DecisionTreeClfPredicted.tolist()
for result in range(len(yTestSetNew)):
    total = total + 1
    if yTestSetNew[result] == DecisionTreeClfPredictedNew[result]:
        correct = correct + 1
print(str(correct)+" correct predictions out of "+str(total)+" for DecisionTree")
print("The DecisionTree percentage of the correct predictions is: "+str(correct/total))

# Setting up SVM and obtaining results
from sklearn import svm
SvmClf     = Pipeline([('clf', svm.SVC(probability=True)), ])
SvmClf.fit(XTrainSet, yTrainSet)
SvmClfScore = cross_val_score(SvmClf, XTrainSet, yTrainSet, cv=2)
print("The 10 fold cross validation score for SVM is : "+str(SvmClfScore))
print("SVM: %0.2f accuracy with a standard deviation of %0.2f" % (SvmClfScore.mean(), SvmClfScore.std()))
SvmClfPredicted = SvmClf.predict(XTestSet)
print("SVM: Classification report:")
print(metrics.classification_report(yTestSet.to_numpy().tolist(), DecisionTreeClfPredicted.tolist()))
correct = 0
total = 0
SvmClfPredictedNew = SvmClfPredicted.tolist()
for result in range(len(yTestSetNew)):
    total = total + 1
    if yTestSetNew[result] == SvmClfPredictedNew[result]:
        correct = correct + 1
print(str(correct)+" correct predictions out of "+str(total)+" for SVM")
print("The SVM percentage of the correct predictions is: "+str(correct/total))

################################## C(3) End

################################## C(4) Start

# Creating prediction probabilities
randomProbs = [0 for _ in range(len(yTestSet))]
multinomialNBClfPredictedProb = multinomialNBClf.predict_proba(XTestSet)
DecisionTreeClfPredictedProb = DecisionTreeClf.predict_proba(XTestSet)
SvmClfPredictedProb = SvmClf.predict_proba(XTestSet)

# Taking only positive predictions
multinomialNBClfPredictedProb = multinomialNBClfPredictedProb[:,1]
DecisionTreeClfPredictedProb = DecisionTreeClfPredictedProb[:,1]
SvmClfPredictedProb = SvmClfPredictedProb[:,1]

# Calculating area under ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
randomProbsAuc = roc_auc_score(yTestSet, randomProbs)
multinomialNBClfPredictedProbAuc = roc_auc_score(yTestSet, multinomialNBClfPredictedProb)
DecisionTreeClfPredictedProbAuc = roc_auc_score(yTestSet, DecisionTreeClfPredictedProb)
SvmClfPredictedProbAuc = roc_auc_score(yTestSet, SvmClfPredictedProb)
print("multinomialNB AUROC = "+str(multinomialNBClfPredictedProbAuc))
print("DecisionTree AUROC = "+str(DecisionTreeClfPredictedProbAuc))
print("Svm AUROC = "+str(SvmClfPredictedProbAuc))

# Calculating FPR and TPR
randomProbsFpr, randomProbsTpr, _ = roc_curve(yTestSet, randomProbs)
multinomialNBClfFpr, multinomialNBClfTpr, _ = roc_curve(yTestSet, multinomialNBClfPredictedProb)
DecisionTreeClfFpr, DecisionTreeClfTpr, _ = roc_curve(yTestSet, DecisionTreeClfPredictedProb)
SvmClfFpr, SvmClfTpr, _ = roc_curve(yTestSet, SvmClfPredictedProb)

# Plotting ROC curve
plt.plot(randomProbsFpr, randomProbsTpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % randomProbsAuc)
plt.plot(multinomialNBClfFpr, multinomialNBClfTpr, marker='.', label='multinomialNB (AUROC = %0.3f)' % multinomialNBClfPredictedProbAuc)
plt.plot(DecisionTreeClfFpr, DecisionTreeClfTpr, marker='.', label='DecisionTree (AUROC = %0.3f)' % DecisionTreeClfPredictedProbAuc)
plt.plot(SvmClfFpr, SvmClfTpr, marker='.', label='SVM (AUROC = %0.3f)' % SvmClfPredictedProbAuc)
plt.title('ROC Plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

################################## C(4) End

######################################### Model Evaluation End (C) #####################################################