import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from matplotlib import colors
from pyparsing import nums

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

class SoftmaxCategorical:
    def __init__(self):
        self.act = actSoftmax()
        self.loss = lossCatCrossEnt()
    def forward(self, inputs):
        self.act.forward(inputs)
        self.output = self.act.output
    def calculateLoss(self, yTrue):
        self.lossValue = self.loss.calculate(self.output, yTrue)
    def backward(self, dvalues, yTrue):
        numSample = dvalues.shape[1]
        if len(yTrue.shape) == 2:
            yTrue = np.argmax(yTrue, axis=0)
        self.dinputs = dvalues.copy()
        self.dinputs[yTrue, range(numSample)] -= 1
        self.dinputs = self.dinputs/numSample

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

class optimizerSGD:
    def __init__(self, initialLearningRate=1, decay=0, momentum=0.9):
        self.initialLearningRate = initialLearningRate
        self.learningRate = self.initialLearningRate
        self.decay = decay
        self.step = 0
        self.momentum = momentum
    def updateLearningRate(self):
        ratio = 1/(1 + self.decay*self.step)
        self.learningRate = self.initialLearningRate*ratio
    def updateLayer(self, layer):
        if self.momentum:
            if not hasattr(layer, "weightMomentum"):
                layer.weightMomentum = np.zeros_like(layer.dweights)
                layer.biasMomentum = np.zeros_like(layer.dbiases)
            deltaWeight = -self.learningRate*layer.dweights + self.momentum*layer.weightMomentum
            deltaBias = -self.learningRate*layer.dbiases + self.momentum*layer.biasMomentum
            layer.weightMomentum = deltaWeight
            layer.biasMomentum = deltaBias
        else:
            deltaWeight = -self.learningRate*layer.dweights
            deltaBias = -self.learningRate*layer.dbiases 
        layer.weights += deltaWeight
        layer.biases += deltaBias
    def updateStep(self):
        self.step += 1

class optimizerAdaGrad:
    def __init__(self, initialLearningRate=1, decay=0, epsilon=1e-7):
        self.initialLearningRate = initialLearningRate
        self.learningRate = self.initialLearningRate
        self.decay = decay
        self.step = 0
        self.epsilon = epsilon
    def updateLearningRate(self):
        ratio = 1/(1 + self.decay*self.step)
        self.learningRate = self.initialLearningRate*ratio
    def updateLayer(self, layer):
        if not hasattr(layer, "weightCache"):
            layer.weightCache = np.zeros_like(layer.dweights)
            layer.biasCache = np.zeros_like(layer.dbiases)
        layer.weightCache += layer.dweights**2
        layer.biasCache += layer.dbiases**2
        deltaWeight = -self.learningRate*layer.dweights/(np.sqrt(layer.weightCache) + self.epsilon)
        deltaBias = -self.learningRate*layer.dbiases/(np.sqrt(layer.biasCache) + self.epsilon)
        layer.weights += deltaWeight
        layer.biases += deltaBias
    def updateStep(self):
        self.step += 1

class optimizerRMSProp:
    def __init__(self, initialLearningRate=0.0001, decay=0, epsilon=1e-7, rho=0.9):
        self.initialLearningRate = initialLearningRate
        self.learningRate = self.initialLearningRate
        self.decay = decay
        self.step = 0
        self.epsilon = epsilon
        self.rho = rho
    def updateLearningRate(self):
        ratio = 1/(1 + self.decay*self.step)
        self.learningRate = self.initialLearningRate*ratio
    def updateLayer(self, layer):
        if not hasattr(layer, "weightCache"):
            layer.weightCache = np.zeros_like(layer.dweights)
            layer.biasCache = np.zeros_like(layer.dbiases)
        layer.weightCache = self.rho*layer.weightCache + (1-self.rho)*layer.dweights**2
        layer.biasCache = self.rho*layer.biasCache + (1-self.rho)*layer.dbiases**2
        deltaWeight = -self.learningRate*layer.dweights/(np.sqrt(layer.weightCache) + self.epsilon)
        deltaBias = -self.learningRate*layer.dbiases/(np.sqrt(layer.biasCache) + self.epsilon)
        layer.weights += deltaWeight
        layer.biases += deltaBias
    def updateStep(self):
        self.step += 1

class optimizerAdam:
    def __init__(self, initialLearningRate=0.001, decay=0, epsilon=1e-7, beta1=0.9, beta2 = 0.999):
        self.initialLearningRate = initialLearningRate
        self.learningRate = self.initialLearningRate
        self.decay = decay
        self.step = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
    def updateLearningRate(self):
        ratio = 1/(1 + self.decay*self.step)
        self.learningRate = self.initialLearningRate*ratio
    def updateLayer(self, layer):
        if not hasattr(layer, "weightCache"):
            layer.weightMomentum = np.zeros_like(layer.dweights)
            layer.weightCache = np.zeros_like(layer.dweights)
            layer.biasMomentum = np.zeros_like(layer.dbiases)
            layer.biasCache = np.zeros_like(layer.dbiases)
        layer.weightMomentum = self.beta1*layer.weightMomentum + (1-self.beta1)*layer.dweights
        layer.biasMomentum = self.beta1*layer.biasMomentum + (1-self.beta1)*layer.dbiases
        weightMomentumCorrected = layer.weightMomentum/(1-self.beta1**(self.step+1))
        biasMomentumCorrected = layer.biasMomentum/(1-self.beta1**(self.step+1))
        layer.weightCache = self.beta2*layer.weightCache + (1-self.beta2)*layer.dweights**2        
        layer.biasCache = self.beta2*layer.biasCache + (1-self.beta2)*layer.dbiases**2
        weightCacheCorrected = layer.weightCache/(1-self.beta2**(self.step+1))
        biasCacheCorrected = layer.biasCache/(1-self.beta2**(self.step+1))
        deltaWeight = -self.learningRate*weightMomentumCorrected/(np.sqrt(weightCacheCorrected) + self.epsilon)
        deltaBias = -self.learningRate*biasMomentumCorrected/(np.sqrt(biasCacheCorrected) + self.epsilon)
        layer.weights += deltaWeight
        layer.biases += deltaBias
    def updateStep(self):
        self.step += 1