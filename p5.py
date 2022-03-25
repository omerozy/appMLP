import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import pprint
import math

nnfs.init()
class layerDense:                               # https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=4
    def __init__(self, numNeurons, numInputs):
        self.weights = 0.1*np.random.randn(numNeurons, numInputs)
        self.biases = np.zeros((numNeurons, 1))
    def forward(self, inputs):
        self.output = np.dot(self.weights, inputs) + self.biases

class  actReLu:                                 # rectified linear activation function
    def forward(self, inputs):                  # https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5
        self.output = np.maximum(0, inputs)

class actSoftmax:                               # softmax activation function
    def forward(self, inputs):                  # https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6
         expValues = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
         probs = expValues / np.sum(expValues, axis=0, keepdims=True)
         self.output = probs
class loss:
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss
class lossCatEnt(loss):                         # categorical cross-entropy
    def forward(self, yPred, yTrue):            # https://www.youtube.com/watch?v=levekYbxauw&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=8
        samples = np.shape(yPred)[1]
        yPredClip = np.clip(yPred, 1e-7, 1-1e-7)        
        if len(yTrue.shape) == 1:
            print(np.shape(yTrue))
            print(np.shape(yPredClip))
            correctConfidences = yPredClip[yTrue, range(samples)]
        else:
            correctConfidences = np.sum(yPredClip*yTrue, axis=0)
        negLogLikelihoods = -np.log(correctConfidences)
        return negLogLikelihoods

X, y = spiral_data(samples=100, classes=3)
X = np.array(X).T
y = np.array(y).T

layer1 = layerDense(3, 2)
act1 = actReLu()

layer2 = layerDense(3, 3)
act2 = actSoftmax()

layer1.forward(X)
act1.forward(layer1.output)

layer2.forward(act1.output)
act2.forward(layer2.output)

pprint.pprint(act2.output[0:3, 0:3])

lossFunc = lossCatEnt()
loss = lossFunc.calculate(act2.output, y)
print("Loss:", loss)