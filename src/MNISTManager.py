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

    def displayLabelStats(self):
        def _displayLabelStatByDatasetType(datasetType): #datasetType = "training" or "testing"
            data = self.trainingLabels if datasetType == "training" else self.testingLabels
            print(f"Occurences in {datasetType} dataset :")
            for i in range(10):
                print(f"{i} -> {np.count_nonzero(data == i)}")
        _displayLabelStatByDatasetType("training")
        _displayLabelStatByDatasetType("testing")

    def displayDigitsMean(self):
        def _displayDigitsMeanByDatasetType(datasetType): #datasetType = "training" or "testing"
            data = self.trainingLabels if datasetType == "training" else self.testingLabels
            greyAdditions = [0.0] * 10
            for i in range(len(data)):
                greyAdditions[self.trainingLabels[i]] += np.mean(self.trainingImages[i])
            print(f"Digits mean in {datasetType} dataset :")
            for i in range(10):
                print(f"{i} -> {greyAdditions[i] / np.count_nonzero(data == i)}")
        _displayDigitsMeanByDatasetType("training")
        _displayDigitsMeanByDatasetType("testing")

    def show(self, index, datasetType): #datasetType = "training" or "testing"
        image = np.asarray(self.trainingImages[index]).squeeze() if datasetType == "training" else  np.asarray(self.testingImages[index]).squeeze()
        plt.imshow(image)
        plt.show()

    