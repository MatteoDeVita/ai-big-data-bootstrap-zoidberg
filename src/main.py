from MNISTMatrix import MNISTMatrix
import os

matrix = MNISTMatrix( os.path.dirname(os.path.abspath(__file__)) + "/../datasets/train-images-idx3-ubyte.gz")
matrix.show(2)
