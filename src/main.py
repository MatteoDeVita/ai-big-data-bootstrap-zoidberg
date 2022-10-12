from KNNManager import KNNManager

knnManager = KNNManager()
knnManager.train(100)
for i in range(10000):
    knnManager.guessDigit(3, i, 1000)