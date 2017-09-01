import numpy as np
import os
import os.path
import sys
import time
import progressbar
import ml_metrics as metrics
import imageUtilities

queryDir = input('Query directory: ')
databaseDir = input('Database directory: ')

i = 0

for root, dirs, files in os.walk('./' + queryDir):
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for queryFile in files:
            relevantFiles = []
            retrievedFiles = imageUtilities.initList(3)
            if queryFile.endswith('.vc'):
                fileName = int(queryFile[7:-4])
                fileMod = int(fileName) % 4
                queryFeature = np.loadtxt(
                    './' + queryDir + '/' + queryFile, delimiter=',')
                for j in range(4):
                    if j < fileMod:
                        relevantFiles.append(fileName - (fileMod - j))
                    elif j > fileMod:
                        relevantFiles.append(fileName + (j - fileMod))
                for root, dirs, files in os.walk('./' + databaseDir):
                    with progressbar.ProgressBar(max_value=len(files)) as bar2:
                        j = 0
                        for databaseFile in files:
                            if databaseFile.endswith('.vc') and databaseFile != queryFile:
                                databaseFeature = np.loadtxt(
                                    './' + databaseDir + '/' + databaseFile, delimiter=',')
                                dist = np.linalg.norm(
                                    databaseFeature - queryFeature)
                                if(dist < imageUtilities.getMaxNeighbour(retrievedFiles)):
                                    retrievedFiles = imageUtilities.addToNeighbours(
                                        neighbour={'filename': databaseFile[7:-3], 'distance': dist}, neighbour_list=retrievedFiles)
                                j += 1
                                bar2.update(j)
                        retrievedFeatures = []
                        for k in retrievedFiles:
                            retrievedFeatures.append(int(k['filename']))
                        apk = metrics.apk(actual=relevantFiles,
                                          predicted=retrievedFeatures, k=3)
                        print('\n')
                        print(relevantFiles)
                        print(retrievedFeatures)
                        print('APK: ', apk)
            i += 1
            bar.update(i)
