import gzip
from pickletools import uint8
import numpy as np
import matplotlib.pyplot as plt

class MNISTMatrix():
    def __init__(self, path):
        zipBuffer = gzip.open(path,'r') #60000 images of 28*28 pixels
        zipBuffer.read(16)
        self.data = np.frombuffer(zipBuffer.read(28 * 28 * 60000), dtype=np.uint8).astype(np.float32)
        self.data = self.data.reshape(28, 28, 60000, 1)


    def show(self, index):
        image = np.asarray(self.data[index]).squeeze()
        plt.imshow(image)
        plt.show()

