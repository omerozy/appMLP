import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

# CREATE CLASSES TO BE USED FOR DEFINING LAYER PROPERTIES (WEIGHT, BIAS, ACTIVATION FUNCTION, LOSS FUNCTION)

class layerDense:                               # https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=4
    def __init__(self, numNeurons, numInputs):  
        self.weights = 0.1*np.random.randn(numNeurons, numInputs) # it is desired for weights to have values between -1 and 1
        self.biases = np.zeros((numNeurons, 1)) # biases are used to prevent neurons from producing zeros
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.weights, inputs) + self.biases
    def backward(self, dvalues):
        # Gradients on biases
        self.dbiases = np.sum(dvalues, axis = 1, keepdims = True)
        # Gradients on weights
        self.dweights = np.dot(self.inputs, dvalues.T)
        self.dweights = self.dweights.T
        # Gradients on inputs
        self.dinputs = np.dot(self.weights.T, dvalues)

class  actReLu:                                 # rectified linear activation function
    def forward(self, inputs):
        self.inputs = inputs                  # https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class actLin:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

class actSoftmax:                               # softmax activation function (exponentiation and normalization for returning correctness probability distribution fror each sample)
    def forward(self, inputs):                  # https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6
         expValues = np.exp(inputs - np.max(inputs, axis=0, keepdims=True)) 
         probs = expValues / np.sum(expValues, axis=0, keepdims=True)   # sum through row (in other words, sum the column)
         self.output = probs
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        numSamples = self.output.shape[1]
        for sampleIdx in range (numSamples):
            singleOutput = self.output[:, [sampleIdx]]
            singleDVlaues = dvalues[:, [sampleIdx]]
            jacobian = np.diagflat(singleOutput) - np.dot(singleOutput, singleOutput.T)
            self.dinputs[:, [sampleIdx]] = np.dot(jacobian.T, singleDVlaues)

class lossCatCrossEnt:                                     
    def forward(self, yPred, yTrue):            #   categorical cross-entropy
        samples = np.shape(yPred)[1]            #   https://www.youtube.com/watch?v=levekYbxauw&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=8
        yPredClip = np.clip(yPred, 1e-7, 1-1e-7)        # crop to prevent log(0) error
        if len(yTrue.shape) == 1:
            correctConfidences = yPredClip[yTrue, range(samples)]
        else:
            correctConfidences = np.sum(yPredClip*yTrue, axis=0)
        negLogLikelihoods = -np.log(correctConfidences)
        return negLogLikelihoods
    def calculate(self, output, y):
        sampleLosses = self.forward(output, y)
        dataLoss = np.mean(sampleLosses)
        return dataLoss
    def backward(self, dvalues, yTrue):
        numSample = dvalues.shape[1]
        numLabel = dvalues.shape[0]
        if len(yTrue.shape) == 1:
            yTrue = np.eye(numLabel)[:, yTrue]
        self.dinputs = -yTrue/dvalues
        self.dinputs = self.dinputs/numSample


