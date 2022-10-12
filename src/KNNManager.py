from MNISTManager import MNISTManager
import os
import pandas as pd
import numpy as np
import sys
from terminaltables import AsciiTable

class KNNManager():
    def __init__(self):
        self.matrices = MNISTManager(
            os.path.dirname(os.path.abspath(__file__)) + "/../datasets/train-images-idx3-ubyte.gz",
            os.path.dirname(os.path.abspath(__file__)) + "/../datasets/train-labels-idx1-ubyte.gz",
            os.path.dirname(os.path.abspath(__file__)) + "/../datasets/t10k-images-idx3-ubyte.gz",
            os.path.dirname(os.path.abspath(__file__)) + "/../datasets/t10k-labels-idx1-ubyte.gz",
        ).getMatrices()

    def _getEuclidianDistance(self, x, y): #get the euclidian distance between 2 matrices
        return np.sqrt(np.sum(x - y)**2);
    
    def _getKNearestNeighbors(self, k, sortedDistances):
        kDistances = sortedDistances[:k]
        return [x[1] for x in kDistances]
    
    def train(self, batchSize):
    
        def _getAccuraryPercentage(correspondingLabels):
            accuracy = 0;
            for i in range(len(correspondingLabels)):
                if self.matrices['trainingLabels'][i] == correspondingLabels[i]:
                    accuracy += 1
            return (accuracy / len(correspondingLabels) * 100)

        kValues = [(3, 0),(5, 0), (7, 0), (9, 0)]  #[(kNeigbors, accuracyPercentage)]

        for kIndex in range(len(kValues)):
            correspondingLabels = [] # for each image, we store is found label
            for i in range(len(self.matrices['trainingImages'][:batchSize])):
                print(f"\t{ round(   ((kIndex / len(kValues) * 100) + ( (i / batchSize) ) / len(kValues) * 100 ), 5   ) }%", end='\r')
                sys.stdout.flush()
                distances = []
                for j in range(len(self.matrices['trainingImages'][:batchSize])):
                    distances.append( ( self._getEuclidianDistance(self.matrices['trainingImages'][i], self.matrices['trainingImages'][j]), self.matrices['trainingLabels'][j]) )
                distances.sort(key = lambda x : x[0])
                kNearestNeighbors = self._getKNearestNeighbors(kValues[kIndex][0], distances)
                correspondingLabels.append( max( kNearestNeighbors, key=kNearestNeighbors.count ) ) #for each image, we adds most propable neigbor for the current value of k
            kValues[kIndex] = (kValues[kIndex][0], _getAccuraryPercentage(correspondingLabels))
        kValuesTable = AsciiTable(
            [
                ["K"] + [x[0] for x in kValues],
                ["Accuracy"] + [str( round( x[1] , 3 ) ) + "%" for x in kValues]
            ]
        )
        print(kValuesTable.table)

    def guessDigit(self, k, testingDigitIndex, batchSize): #batchSize should be around 1000 with a correct k value
       
        distances = []
        for i in range(len(self.matrices['testingImages'][:batchSize])):
            distances.append( (self._getEuclidianDistance( self.matrices['testingImages'][testingDigitIndex], self.matrices['testingImages'][i] ), self.matrices['testingLabels'][i]) )
        distances.sort(key = lambda x : x[0]) #sort by distance
        kNearestNeighbors = self._getKNearestNeighbors(k, distances)
        value = max( kNearestNeighbors, key=kNearestNeighbors.count )
        print(f"Guessed value : {value} | Real value : {self.matrices['testingLabels'][testingDigitIndex]} | correct : {self.matrices['testingLabels'][testingDigitIndex] == value}")                   

            
