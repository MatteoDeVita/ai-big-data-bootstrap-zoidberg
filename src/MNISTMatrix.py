import gzip
from pickletools import uint8
import numpy as np
import matplotlib.pyplot as plt

class MNISTMatrix():
    def __init__(
        self, 
        trainingSetImagesPath,
        trainingSetLabelsPath,
        testingSetImagesPath,
        testingSetLabelsPath
    ):
       self.trainingImages = self._getImageData(trainingSetImagesPath, 60000)
       self.trainingLabels = self._getLabelData(trainingSetLabelsPath, 60000)
       self.testingImages = self._getImageData(testingSetImagesPath, 10000)
       self.testingLabels = self._getLabelData(testingSetLabelsPath, 10000)

       print(self.trainingLabels)
       

    def _getImageData(self, path, size):
        fileBuffer = gzip.open(path,'r')
        fileBuffer.read(16)
        images = np.frombuffer(fileBuffer.read(28 * 28 * size), dtype=np.uint8).astype(np.float32) #60000 images of 28*28 pixels
        return images.reshape(size, 28, 28, 1)

    def _getLabelData(self, path, size):
        fileBuffer = gzip.open(path,'r')      
        fileBuffer.read(8)
        return np.frombuffer(fileBuffer.read(size), dtype=np.uint8).astype(np.uint64)

    def show(self, index, datasetType): #datasetType = "training" or "testing"
        image = np.asarray(self.trainingImages[index]).squeeze() if datasetType == "training" else  np.asarray(self.testingImages[index]).squeeze()
        plt.imshow(image)
        plt.show()