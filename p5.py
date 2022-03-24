import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import pprint
import math

nnfs.init()
class layerDense:
    def __init__(self, numNeurons, numInputs):
        self.weights = 0.1*np.random.randn(numNeurons, numInputs)
        self.biases = np.zeros((numNeurons, 1))
    def forward(self, inputs):
        self.output = np.dot(self.weights, inputs) + self.biases

class  actReLu:                                 # rectified linear activation function
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class actSoftmax:                               # softmax activation function
    def forward(self, inputs):
         expValues = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
         probs = expValues / np.sum(expValues, axis=0, keepdims=True)
         self.output = probs


X, y = spiral_data(samples=100, classes=3)
X = np.array(X).T

layer1 = layerDense(3, 2)
act1 = actReLu()

layer2 = layerDense(3, 3)
act2 = actSoftmax()

layer1.forward(X)
act1.forward(layer1.output)

layer2.forward(act1.output)
act2.forward(layer2.output)

pprint.pprint(act2.output[0:3, 0:5])