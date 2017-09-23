import os
import os.path
import sys
import time
from multiprocessing import Process
from operator import itemgetter

import ml_metrics as metrics
import numpy as np
import progressbar
from joblib import Parallel, delayed
from imageDistortion import imageDirList

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


def compareMatrix(a, b):
    nrm = np.linalg.norm(np.subtract(a, b))
    return nrm


def worker(queryImage, j, databaseDir):
    databaseImage = np.loadtxt(
        './' + databaseDir + '/' + str(j) + '.vc', delimiter=',')
    distance = compareMatrix(databaseImage, queryImage)
    return {'filename': j, 'distance': distance}


imageDirList = imageDirList

if __name__ == '__main__':
    for compare in imageDirList:
        print(compare)
        queryDir = compare + 'Features'
        databaseDir = 'holidaysFeatures'

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
            distance = Parallel(n_jobs=-1)(delayed(worker)(
                queryImage, referenceList[j], databaseDir) for j in range(len(referenceList)))
            nearestNeighbours = sorted(distance, key=itemgetter('distance'))[
                :len(relevantList)]
            retrievedList = []
            for k in nearestNeighbours:
                retrievedList.append(int(k['filename']))
            apk = metrics.apk(actual=relevantList, predicted=retrievedList)
            listOfReleventList.append(relevantList)
            listOfRetrievedList.append(retrievedList)
        mapk = metrics.mapk(actual=listOfReleventList,
                            predicted=listOfRetrievedList)
        print(mapk)
        text = '\nMAPK for ' + queryDir + ' against ' + \
            databaseDir + ' is ' + str(mapk)
        file = open("mapk.txt", "a")
        file.write(text)
        file.close()
