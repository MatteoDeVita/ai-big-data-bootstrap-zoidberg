from MNISTManager import MNISTManager
import os

mnistData = MNISTManager(
    os.path.dirname(os.path.abspath(__file__)) + "/../datasets/train-images-idx3-ubyte.gz",
    os.path.dirname(os.path.abspath(__file__)) + "/../datasets/train-labels-idx1-ubyte.gz",
    os.path.dirname(os.path.abspath(__file__)) + "/../datasets/t10k-images-idx3-ubyte.gz",
    os.path.dirname(os.path.abspath(__file__)) + "/../datasets/t10k-labels-idx1-ubyte.gz",
)

print(mnistData.getLabelStats("training"))
print(mnistData.getLabelStats("testing"))
print(mnistData.getDigitsMean("training"))
print(mnistData.getDigitsMean("testing"))
