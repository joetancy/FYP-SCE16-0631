import numpy as np
import os
import os.path
import sys
import time
import progressbar
import ml_metrics as metrics
import imageUtilities


def buildAllList(queryDir):
    allFileList = []
    for root, dirs, files in os.walk('./' + queryDir):
        for file in files:
            allFileList.append(int(file[:-3]))
    return allFileList


def buildQueryList(queryDir):
    queryList = []
    for root, dirs, files in os.walk('./' + queryDir):
        for file in files:
            if int(file[:-3]) % 100 == 0:
                queryList.append(int(file[:-3]))
    return queryList


def buildRelevantList(queryDir, file):
    relevantList = []
    allList = buildAllList(queryDir)
    length = len(allList)
    index = 0
    for i in range(length):
        if allList[i] == file:
            index = i + 1
            break
    while index < length:
        if (allList[index] % 100) != 0:
            relevantList.append(allList[index])
        else:
            break
        index += 1
    return relevantList


def buildReferenceList(queryDir):
    referenceList = [x for x in buildAllList(
        queryDir) if x not in buildQueryList(queryDir)]
    return referenceList


queryDir = input('Query directory: ')
databaseDir = input('Database directory: ')

queryList = buildQueryList(queryDir)
referenceList = buildReferenceList(queryDir)

listOfReleventList = []
listOfRetrievedList = []

bar = progressbar.ProgressBar(redirect_stdout=True)
for i in bar(range(len(queryList))):
    queryImage = np.loadtxt('./' + queryDir + '/' +
                            str(queryList[i]) + '.vc', delimiter=',')
    relevantList = buildRelevantList(queryDir, queryList[i])
    nearestNeighbours = imageUtilities.initList(len(relevantList))
    for j in range(len(referenceList)):
        databaseImage = np.loadtxt(
            './' + databaseDir + '/' + str(referenceList[j]) + '.vc', delimiter=',')
        distance = np.linalg.norm(databaseImage - queryImage)
        if (distance < imageUtilities.getMaxNeighbour(nearestNeighbours)):
            nearestNeighbours = imageUtilities.addToNeighbours(
                neighbour={'filename': referenceList[j], 'distance': distance}, neighbour_list=nearestNeighbours)
    retrievedList = []
    for k in nearestNeighbours:
        retrievedList.append(int(k['filename']))
    apk = metrics.apk(actual=relevantList, predicted=retrievedList)
    listOfReleventList.append(relevantList)
    # print(relevantList)
    listOfRetrievedList.append(retrievedList)
    # print(retrievedList)
    # print('APK: ', apk)
mapk = metrics.mapk(actual=listOfReleventList, predicted=listOfRetrievedList)
print('MAPK for', queryDir, 'against', databaseDir, 'is', mapk)
