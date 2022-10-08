import gzip
from pickletools import uint8
import numpy as np
import matplotlib.pyplot as plt

class MNISTManager():
    def __init__(
        self, 
        trainingSetImagesPath,
        trainingSetLabelsPath,
        testingSetImagesPath,
        testingSetLabelsPath
    ):
        #60000 training data and 10000 testing data
        self.trainingImages = self._getImageData(trainingSetImagesPath, 60000)
        self.trainingLabels = self._getLabelData(trainingSetLabelsPath, 60000)
        self.testingImages = self._getImageData(testingSetImagesPath, 10000)
        self.testingLabels = self._getLabelData(testingSetLabelsPath, 10000)      
       

    def _getImageData(self, path, size):
        fileBuffer = gzip.open(path,'r')
        fileBuffer.read(16)
        images = np.frombuffer(fileBuffer.read(28 * 28 * size), dtype=np.uint8).astype(np.float32) #60000 images of 28*28 pixels
        return images.reshape(size, 784, 1)

    def _getLabelData(self, path, size):
        fileBuffer = gzip.open(path,'r')      
        fileBuffer.read(8)
        return np.frombuffer(fileBuffer.read(size), dtype=np.uint8).astype(np.uint64)
    
    #datasetType = "training" or "testing"
    def getLabelStats(self, datasetType): return [np.count_nonzero(self.trainingLabels == i) for i in range(10)] if datasetType == "training" else [np.count_nonzero(self.testingLabels == i) for i in range(10)]

    #datasetType = "training" or "testing"
    def getDigitsMean(self, datasetType):
        (labels, images) = (self.trainingLabels, self.trainingImages) if datasetType == "training" else (self.testingLabels, self.testingImages)
        greyMeans = [0.0] * 10
        for i in range(len(images)):
            greyMeans[labels[i]] += np.mean(images[i])
        return [greyMeans[i] / np.count_nonzero(labels == i) for i in range(10)] #divide each value by the total number of the digit to get the mean value
        
    def show(self, index, datasetType): #datasetType = "training" or "testing"
        image = np.asarray(self.trainingImages[index]).squeeze() if datasetType == "training" else  np.asarray(self.testingImages[index]).squeeze()
        plt.imshow(image)
        plt.show()

    def displayLabelStatsGraph(self, datasetType):
        _, ax = plt.subplots()
        ax.bar(
            range(10),
            self.getDigitsMean(datasetType),
            label = [str(x) for x in range(10)]            
        )
        plt.show()
