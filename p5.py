import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import pprint


nnfs.init()

X, y = spiral_data(100, 3)
X = np.array(X).T

class layerDense:
    def __init__(self, numNeurons, numInputs):
        self.weights = 0.1*np.random.randn(numNeurons, numInputs)
        self.biases = np.zeros((numNeurons, 1))
    def forward(self, inputs):
        self.output = np.dot(self.weights, inputs) + self.biases

class  actReLu:                                 # rectified linear
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = layerDense(5, 2) 
act1 = actReLu()

layer1.forward(X)
pprint.pprint(layer1.output)
act1.forward(layer1.output)
pprint.pprint(act1.output)