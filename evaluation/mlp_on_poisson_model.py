import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from predictor.neural_network import *

if __name__ == '__main__':
    ann = NeuralNetwork(2, 100, 0.05)
