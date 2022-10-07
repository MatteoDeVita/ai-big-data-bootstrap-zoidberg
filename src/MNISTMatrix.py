import gzip
from pickletools import uint8
import numpy as np
import matplotlib.pyplot as plt

class MNISTMatrix():
    def __init__(self, path):
        fileBuffer = gzip.open(path,'r') #60000 images of 28*28 pixels
        fileBuffer.read(16)
        dataBuffer = fileBuffer.read(28 * 28 * 60000)
        self.data = np.frombuffer(dataBuffer, dtype=np.uint8).astype(np.float32)
        self.data = self.data.reshape(60000, 28, 28, 1)


    def show(self, index):
        image = np.asarray(self.data[index]).squeeze()
        plt.imshow(image)
        plt.show()

