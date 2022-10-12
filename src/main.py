from KNNManager import KNNManager

knnManager = KNNManager()
knnManager.train(100)
knnManager.guessDigit(3, 569, 1000)