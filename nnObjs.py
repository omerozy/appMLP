import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from matplotlib import colors
from pyparsing import nums
import copy

nnfs.init()

# CREATE CLASSES TO BE USED FOR DEFINING LAYER PROPERTIES (WEIGHT, BIAS, ACTIVATION FUNCTION, LOSS FUNCTION)

class layerDense:                               # https://www.youtube.com/watch?v=TEWy9vZcxW4&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=4

    def __init__(self, numNeurons, numInputs, weightRegularizerL1=0, biasRegularizerL1=0, weightRegularizerL2=0, biasRegularizerL2=0):  
        self.weight = 0.1*np.random.randn(numNeurons, numInputs) # it is desired for weights to have values between -1 and 1
        self.bias = np.zeros((numNeurons, 1)) # biases are used to prevent neurons from producing zeros
        self.weightRegularizerL1 = weightRegularizerL1
        self.biasRegularizerL1 = biasRegularizerL1
        self.weightRegularizerL2 = weightRegularizerL2
        self.biasRegularizerL2 = biasRegularizerL2

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.weight, inputs) + self    .bias

    def backward(self, dvalues):
        # Gradients on biases
        self.dbias = np.sum(dvalues, axis = 1, keepdims = True)
        dL1 = np.ones_like(self.bias)
        dL1[self.bias < 0] = -1
        dL2 = 2*self.bias
        self.dbias = self.dbias + self.biasRegularizerL1*dL1 + self.biasRegularizerL2*dL2

        # Gradients on weights
        self.dweight = np.dot(self.inputs, dvalues.T)
        self.dweight = self.dweight.T
        dL1 = np.ones_like(self.weight)
        dL1[self.weight < 0] = -1
        dL2 = 2*self.weight
        self.dweight = self.dweight + self.weightRegularizerL1*dL1 + self.weightRegularizerL2*dL2

        # Gradients on inputs
        self.dinputs = np.dot(self.weight.T, dvalues)

class layerDropout:
    
    def __init__(self, dropoutRate):
        self.rate = 1-dropoutRate
    
    def forward(self, inputs):
        self.inputs = inputs
        if self.rate != 1:
            self.binaryMask = np.random.binomial(1, self.rate, size=inputs.shape)/self.rate
            self.output = inputs*self.binaryMask
        else:
            self.output = inputs

    def backward(self, dvalues):
        if self.rate != 1:
            self.dinputs = dvalues*self.binaryMask
        else:
            self.dinputs = dvalues

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

    def calculateOutputLoss(self, output, y):
        lossesPerSample = self.forward(output, y)
        outputLoss = np.mean(lossesPerSample)
        return outputLoss

    def calculateRegLoss(self, layer):
        regLoss = 0
        regLoss += layer.weightRegularizerL1   *np.sum(np.abs(layer.weight)             )
        regLoss += layer.biasRegularizerL1     *np.sum(np.abs(layer.bias)               )
        regLoss += layer.weightRegularizerL2   *np.sum(layer.weight*layer.weight        )
        regLoss += layer.biasRegularizerL2     *np.sum(layer.bias*layer.bias            )
        return regLoss

    def backward(self, dvalues, yTrue):
        numSample = dvalues.shape[1]
        numLabel = dvalues.shape[0]
        if len(yTrue.shape) == 1:
            yTrue = np.eye(numLabel)[:, yTrue]
        self.dinputs = -yTrue/dvalues
        self.dinputs = self.dinputs/numSample

class SoftmaxCategorical:

    def __init__(self):
        self.act = actSoftmax()
        self.loss = lossCatCrossEnt()

    def forward(self, inputs):
        self.act.forward(inputs)
        self.output = self.act.output

    def calculateOutputLoss(self, yTrue):
        self.outputLoss = self.loss.calculateOutputLoss(self.output, yTrue)

    def calculateRegLoss(self, layer):
        return self.loss.calculateRegLoss(layer)

    def backward(self, dvalues, yTrue):
        numSample = dvalues.shape[1]
        if len(yTrue.shape) == 2:
            yTrue = np.argmax(yTrue, axis=0)
        self.dinputs = dvalues.copy()
        self.dinputs[yTrue, range(numSample)] -= 1
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
                layer.weightMomentum = np.zeros_like(layer.dweight)
                layer.biasMomentum = np.zeros_like(layer.dbias)
            deltaWeight = -self.learningRate*layer.dweight + self.momentum*layer.weightMomentum
            deltaBias = -self.learningRate*layer.dbias + self.momentum*layer.biasMomentum
            layer.weightMomentum = deltaWeight
            layer.biasMomentum = deltaBias
        else:
            deltaWeight = -self.learningRate*layer.dweight
            deltaBias = -self.learningRate*layer.dbias 
        layer.weight += deltaWeight
        layer.bias += deltaBias
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
            layer.weightCache = np.zeros_like(layer.dweight)
            layer.biasCache = np.zeros_like(layer.dbias)
        layer.weightCache += layer.dweight**2
        layer.biasCache += layer.dbias**2
        deltaWeight = -self.learningRate*layer.dweight/(np.sqrt(layer.weightCache) + self.epsilon)
        deltaBias = -self.learningRate*layer.dbias/(np.sqrt(layer.biasCache) + self.epsilon)
        layer.weight += deltaWeight
        layer.bias += deltaBias
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
            layer.weightCache = np.zeros_like(layer.dweight)
            layer.biasCache = np.zeros_like(layer.dbias)
        layer.weightCache = self.rho*layer.weightCache + (1-self.rho)*layer.dweight**2
        layer.biasCache = self.rho*layer.biasCache + (1-self.rho)*layer.dbias**2
        deltaWeight = -self.learningRate*layer.dweight/(np.sqrt(layer.weightCache) + self.epsilon)
        deltaBias = -self.learningRate*layer.dbias/(np.sqrt(layer.biasCache) + self.epsilon)
        layer.weight += deltaWeight
        layer.bias += deltaBias
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
            layer.weightMomentum = np.zeros_like(layer.dweight)
            layer.weightCache = np.zeros_like(layer.dweight)
            layer.biasMomentum = np.zeros_like(layer.dbias)
            layer.biasCache = np.zeros_like(layer.dbias)
        layer.weightMomentum = self.beta1*layer.weightMomentum + (1-self.beta1)*layer.dweight
        layer.biasMomentum = self.beta1*layer.biasMomentum + (1-self.beta1)*layer.dbias
        weightMomentumCorrected = layer.weightMomentum/(1-self.beta1**(self.step+1))
        biasMomentumCorrected = layer.biasMomentum/(1-self.beta1**(self.step+1))
        layer.weightCache = self.beta2*layer.weightCache + (1-self.beta2)*layer.dweight**2        
        layer.biasCache = self.beta2*layer.biasCache + (1-self.beta2)*layer.dbias**2
        weightCacheCorrected = layer.weightCache/(1-self.beta2**(self.step+1))
        biasCacheCorrected = layer.biasCache/(1-self.beta2**(self.step+1))
        deltaWeight = -self.learningRate*weightMomentumCorrected/(np.sqrt(weightCacheCorrected) + self.epsilon)
        deltaBias = -self.learningRate*biasMomentumCorrected/(np.sqrt(biasCacheCorrected) + self.epsilon)
        layer.weight += deltaWeight
        layer.bias += deltaBias
    def updateStep(self):
        self.step += 1