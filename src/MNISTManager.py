import gzip
from pickletools import uint8
from turtle import color
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
        labelStats = self.getLabelStats(datasetType)
        barColors = ["tab:blue" for _ in range(10)]
        barColors[np.argmin(labelStats)] = "tab:red"
        barColors[np.argmax(labelStats)] = "tab:green"
        ax.bar(
            range(10),
            labelStats,
            label = [str(x) for x in range(10)],
            color = barColors,
            alpha = 0.7                      
        )
        for index, value in enumerate(labelStats):
            ax.text(index - 0.415, value - (value * 0.1), str(value), color="black", fontweight='bold', alpha=0.7)
        ax.set_ylabel("Occurences")
        ax.set_title("Occurences by digit")
        plt.show()

    def displayDigitMeansGraph(self, datasetType):
        _, ax = plt.subplots()
        digitMeans = self.getDigitsMean(datasetType)
        barColors = ["tab:blue" for _ in range(10)]
        barColors[np.argmin(digitMeans)] = "tab:red"
        barColors[np.argmax(digitMeans)] = "tab:green"
        ax.bar(
            range(10),
            digitMeans,
            label = [str(x) for x in range(10)],
            color = barColors,
            alpha = 0.7                      
        )
        for index, value in enumerate(digitMeans):
            ax.text(index - 0.415, value - (value * 0.1), str(round(value, 1)), color="black", fontweight='bold', alpha=0.7)
        ax.set_ylabel("Means")
        ax.set_title("Means by digit")
        plt.show()
