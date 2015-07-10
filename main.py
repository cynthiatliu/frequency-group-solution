# Main - Frequency
# Virtually identical in format to main-presence, but
# Using frequency arrays instead of presence arrays

import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, cross_validation
from sknn.mlp import Classifier, Layer
from randomCoords import *

#All-incorporating formatting
def format(clf, toFit):
    #Ask for the number of points in 1 cluster
    goodInput = False
    while goodInput == False:
        try:
            groupSize = int(input("How many points would you like to have in one cluster? "))
            testPositive = math.sqrt(groupSize-1)
            goodInput = True
        except:
            print ("Please input a valid value (numerical value larger than or equal to 1)")
            
    totalPtCount = len(toFit)/7 #Total number of points per structure
    pointLabels = np.zeros([7*math.ceil(totalPtCount/groupSize),groupSize])
    strucCounter = 0 #Number of structures we've been through
    while strucCounter < 7:
        count = 0 #basic counter - number of times we went through the loop
        numGroups = 0 #Number of completed groups within a structure
    
        while numGroups < math.ceil(totalPtCount)/groupSize:
            count = 0
            while count < groupSize and count < totalPtCount - numGroups*groupSize:
                index = strucCounter*totalPtCount + numGroups*groupSize + count
                pointLabels[strucCounter*len(pointLabels)/7+numGroups][count] = clf.predict(np.array([toFit[index]]))
                count += 1
    
            #print("Next group!")
            numGroups += 1
    
        #print ("Onto the next structure!")
        strucCounter += 1
        
    return pointLabels, groupSize

#Training-specific formatting
def formatTrain(clf, toFit):
    trPointLabels, ptsPerGroup = format(clf, toFit)
    print (trPointLabels.shape)
    totalPtCount = len(toFit)/7 #Total number of points per structure
        
    #Labeling groups of points to complete the training set
    targetVariables = np.zeros([len(trPointLabels)])
    count = 0
    for labelSet in trPointLabels:
        groupLabel = math.floor(count/math.ceil((totalPtCount/ptsPerGroup)))
        targetVariables[count] = groupLabel
        count += 1
        
    return trPointLabels, targetVariables

#Formatting the presence array
def formatFreq(arr, labels):
    count = 0
    for group in labels:
        group = [int(x+.00001) for x in group]
        for num in group:
            arr[count][num] += 1
        count += 1
        
    return arr

#Main - neural net for individual points, forest for presence arrays
def main():
    
    rawTrain = genCoords()
    rawX_train = buildTrain(rawTrain)[:,0:9]
    rawY_train = buildTrain(rawTrain)[:,-1]
    
    rawTest1 = genCoords()
    rawX_test1 = buildTest(rawTest1)[:] #Used in clf1 to make individual point predictions, which are grouped together in training group predictions
    rawTest2 = genCoords()
    rawX_test2 = buildTest(rawTest2)[:] #Used for actual testing
    print ("Onion - We've generated all points!")
    
    #Define and train the individual-point neural net - b/c it's accurate
    clf1 = Classifier(
        layers=[
            Layer("Maxout", units=10, pieces=2),
            Layer("Maxout", units=10, pieces=2),
            Layer("Maxout", units=10, pieces=2),
            Layer("Maxout", units=10, pieces=2),      
            Layer("Softmax")],
        learning_rate=0.001,
        n_iter=25)
    clf1.fit(rawX_train, rawY_train)
    print ("Beacon - We've defined and trained the individual point neural net!")
    
    #Formatting the training data
    medX_train, Y_train = formatTrain(clf1, rawX_test1) #Get it...raw, medium, well done? Ok, that was a horrid pun
    X_train = np.zeros([len(medX_train),7])
    X_train = formatFreq(X_train, medX_train)
    
    #Formatting the testing data
    medX_test, groupSize = format(clf1, rawX_test2)
    
    #Reading the label groups to file
    labelGroups = open("Label Groups 2.txt", 'w')
    counter = 1
    for group in medX_test:
        labelGroups.write(str(counter))
        labelGroups.write(str(group))
        labelGroups.write("\n\n")
        counter += 1
        
    X_test = np.zeros([len(medX_test),7])
    X_test = formatFreq(X_test, medX_test)
    
    #Reading presence groups to file
    freqGroups = open("Frequency Groups 1.txt", 'w')
    counter = 1
    for group in X_test:
        freqGroups.write(str(counter))
        freqGroups.write(str(group))
        freqGroups.write("\n\n")
        counter += 1
    
    print ("Jasmine Orange - We've formatted all data and read the test data to file!")
    
    #Building the random forest classifier
    clf2 = RandomForestClassifier(n_estimators=300)
    clf2 = clf2.fit(X_train, Y_train)
    
    Y_test = clf2.predict(X_test)
    results = open("results 2.txt", 'w')
    for i in range(len(Y_test)/5):
        results.write("{}\t{}\t{}\t{}\t{}".format(Y_test[5*i], Y_test[5*i+1], Y_test[5*i+2], Y_test[5*i+3], Y_test[5*i+4]))
        results.write("\n")
        if (i%(math.ceil(1200/groupSize)) == (math.ceil(1200/groupSize)-1)): results.write("\n")    
        
main()
