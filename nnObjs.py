import numpy as np
import nnfs
from nnfs.datasets import spiral_data, sine_data
import matplotlib.pyplot as plt
from matplotlib import colors
from pyparsing import nums
import pickle
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

    def forward(self, inputs, training=False):

        self.inputs = inputs
        self.output = np.dot(self.weight, inputs) + self.bias

    def backward(self, dvalues):
        
        # Gradients on biases
        self.dbias = np.sum(dvalues, axis = 1, keepdims = True)

        if self.biasRegularizerL1 > 0:
            dL1 = np.ones_like(self.bias)
            dL1[self.bias < 0] = -1
            self.dbias += self.biasRegularizerL1*dL1

        if self.biasRegularizerL2 > 0:
            dL2 = 2*self.bias
            self.dbias += self.biasRegularizerL2*dL2

        # Gradients on weights
        self.dweight = np.dot(self.inputs, dvalues.T)
        self.dweight = self.dweight.T

        if self.weightRegularizerL1 > 0:
            dL1 = np.ones_like(self.weight)
            dL1[self.weight < 0] = -1
            self.dweight += self.weightRegularizerL1*dL1

        if self.weightRegularizerL2 > 0:
            dL2 = 2*self.weight
            self.dweight += self.weightRegularizerL2*dL2

        # Gradients on inputs
        self.dinputs = np.dot(self.weight.T, dvalues)

    def getParams(self):
        return self.weight, self.bias

    def setParams(self, weight, bias):
        self.weight = weight
        self.bias = bias

class layerDropout:
    
    def __init__(self, dropoutRate):
        self.rate = 1-dropoutRate
    
    def forward(self, inputs, training):

        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
        else:
            if self.rate != 1:
                self.binaryMask = np.random.binomial(1, self.rate, size=inputs.shape)/self.rate
                self.output = inputs*self.binaryMask
            else:
                self.output = inputs.copy()

    def backward(self, dvalues):
        if self.rate != 1:
            self.dinputs = dvalues*self.binaryMask
        else:
            self.dinputs = dvalues

class layerInput:
    def forward(self, inputs, training=False):
        self.output = inputs

class  actReLu:                                 # rectified linear activation function

    def forward(self, inputs, training=False):
        self.inputs = inputs                  # https://www.youtube.com/watch?v=gmjzbpSVY1A&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=5
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def prediction(self, outputs):
        return outputs

class actLin:

    def forward(self, inputs, training=False):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    def prediction(self, outputs):
        return outputs

class actSoftmax:                               # softmax activation function (exponentiation and normalization for returning correctness probability distribution fror each sample)

    def forward(self, inputs, training):                  # https://www.youtube.com/watch?v=omz_NdFgWyU&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=6
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

    def prediction(self, outputs):
        return np.argmax(outputs, axis=0, keepdims=True)

class actSigmoid:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1/(1+np.exp(-inputs))
    
    def backward(self, dvalues):
        self.dinputs = self.output*(1-self.output)*dvalues

    def prediction(self, outputs):
        return (outputs>0.5)*1

class accuracy:

    def calculateAccuracy(self, prediction, y):
        comparison  = self.compare(prediction, y)
        acc         = np.mean(comparison)
        self.accumAcc    += np.sum(comparison)
        self.accumCount  += comparison.shape[1] 
        return acc

    def calculateAccumAcc(self):
        acc = self.accumAcc/self.accumCount
        return acc

    def resetAccumAcc(self):
        self.accumAcc     = 0
        self.accumCount   = 0
        
class accuracyCategorical(accuracy):

    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, prediction, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=0)
        return prediction == y

class accuracyRegression(accuracy):

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):

        if self.precision is None or reinit:
            self.precision = np.std(y)/250

    def compare(self, prediction, y):

        return np.absolute(prediction - y) < self.precision

class loss:
    
    def keepTrainableLayers(self, trainableLayers):
        self.trainableLayers = trainableLayers

    def calculateOutputLoss(self, output, y):
        lossPerSample = self.forward(output, y)
        outputLoss = np.mean(lossPerSample)
        self.accumLoss  += np.sum(lossPerSample)
        self.accumCount += len(lossPerSample)
        return outputLoss
        
    def calculateRegLoss(self):
        regLoss = 0
        for layer in self.trainableLayers:
            if layer.weightRegularizerL1 > 0:
                regLoss += layer.weightRegularizerL1   *    np.sum(     np.abs(layer.weight)             )
            if layer.biasRegularizerL1 > 0:
                regLoss += layer.biasRegularizerL1     *    np.sum(     np.abs(layer.bias)               )
            if layer.weightRegularizerL2 > 0:
                regLoss += layer.weightRegularizerL2   *    np.sum(     layer.weight*layer.weight        )
            if layer.biasRegularizerL2 > 0:
                regLoss += layer.biasRegularizerL2     *    np.sum(     layer.bias*layer.bias            )
        return regLoss

    def calculateLoss(self, output, y, includeRegLoss=False):
        if not includeRegLoss:
            return self.calculateOutputLoss(output, y)
        else:
            return self.calculateOutputLoss(output, y), self.calculateRegLoss()

    def calculateAccumLoss(self, *, includeRegLoss=False):
        outputLoss = self.accumLoss/self.accumCount
        if not includeRegLoss:
            return outputLoss
        else:
            return outputLoss, self.calculateRegLoss()

    def resetAccumLoss(self):
        self.accumLoss  = 0
        self.accumCount  = 0

class lossCatCrossEnt(loss):

    def forward(self, yPred, yTrue):            #   categorical cross-entropy
        samples = np.shape(yPred)[1]            #   https://www.youtube.com/watch?v=levekYbxauw&list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3&index=8
        yPredClip = np.clip(yPred, 1e-7, 1-1e-7)        # crop to prevent log(0) error
        if len(yTrue.shape) == 1:
            correctConfidences = yPredClip[yTrue, range(samples)]
        else:
            correctConfidences = np.sum(yPredClip*yTrue, axis=0)
        negLogLikelihoods = -np.log(correctConfidences)
        return negLogLikelihoods

    def backward(self, dvalues, yTrue):
        numSample = dvalues.shape[1]
        numLabel = dvalues.shape[0]
        if len(yTrue.shape) == 1:
            yTrue = np.eye(numLabel)[:, yTrue]
        self.dinputs = -yTrue/dvalues
        self.dinputs = self.dinputs/numSample

class lossBinCrossEnt(loss):

    def forward(self, yPred, yTrue):   

        yPredClip = np.clip(yPred, 1e-7, 1-1e-7) # crop to prevent log(0) error
        sampleLoss = -(yTrue*np.log(yPredClip) + (1-yTrue)*np.log(1-yPredClip))
        sampleLoss = np.mean(sampleLoss, axis=0, keepdims=True)
        return sampleLoss

    def backward(self, dvalues, yTrue):
        numSample = dvalues.shape[1]
        numLabel = dvalues.shape[0]
        dValuesClip = np.clip(dvalues, 1e-7, 1-1e-7)
        self.dinputs = -(yTrue/dValuesClip - (1-yTrue)/(1-dValuesClip))/numLabel
        self.dinputs = self.dinputs/numSample

class lossMeanSquaredError(loss):

    def forward(self, yPred, yTrue):   

        sampleLoss = np.mean((yTrue-yPred)**2, axis=0, keepdims=True)

        return sampleLoss

    def backward(self, dvalues, yTrue):
        numSample = dvalues.shape[1]
        numLabel = dvalues.shape[0]
        self.dinputs = -2*(yTrue-dvalues)/numLabel
        self.dinputs = self.dinputs/numSample

class lossMeanAbsoluteError(loss):

    def forward(self, yPred, yTrue):   

        sampleLoss = np.mean(np.abs(yTrue-yPred), axis=0, keepdims=True)

        return sampleLoss

    def backward(self, dvalues, yTrue):
        numSample = dvalues.shape[1]
        numLabel = dvalues.shape[0]
        self.dinputs = np.sign(yTrue-dvalues)/numLabel
        self.dinputs = self.dinputs/numSample

class SoftmaxCategorical:

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
            layer.weightMomentum    = np.zeros_like(layer.dweight)
            layer.weightCache       = np.zeros_like(layer.dweight)
            layer.biasMomentum      = np.zeros_like(layer.dbias)
            layer.biasCache         = np.zeros_like(layer.dbias)

        layer.weightMomentum    = self.beta1*layer.weightMomentum   + (1-self.beta1)*layer.dweight
        layer.biasMomentum      = self.beta1*layer.biasMomentum     + (1-self.beta1)*layer.dbias

        weightMomentumCorrected = layer.weightMomentum  /(1-self.beta1**(self.step+1))
        biasMomentumCorrected   = layer.biasMomentum    /(1-self.beta1**(self.step+1))

        layer.weightCache       = self.beta2*layer.weightCache  + (1-self.beta2)*layer.dweight**2        
        layer.biasCache         = self.beta2*layer.biasCache    + (1-self.beta2)*layer.dbias**2

        weightCacheCorrected    = layer.weightCache /(1-self.beta2**(self.step+1))
        biasCacheCorrected      = layer.biasCache   /(1-self.beta2**(self.step+1))

        deltaWeight             = -self.learningRate*weightMomentumCorrected    /   (np.sqrt(  weightCacheCorrected  )     + self.epsilon)
        deltaBias               = -self.learningRate*biasMomentumCorrected      /   (np.sqrt(  biasCacheCorrected    )     + self.epsilon)

        layer.weight            += deltaWeight
        layer.bias              += deltaBias

    def updateStep(self):
        self.step += 1

class model:
    def __init__(self):
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

    def set(self, *, loss=None, optimizer=None, accuracy=None):

        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    def establish(self):

        self.layerInput = layerInput()

        numLayer = len(self.layers)
        
        self.trainableLayers = []
        
        for layerIdx in range(numLayer):

            if layerIdx == 0:
                self.layers[layerIdx].prev = self.layerInput
                self.layers[layerIdx].next = self.layers[layerIdx + 1]
            elif layerIdx < numLayer - 1:
                self.layers[layerIdx].prev = self.layers[layerIdx - 1]
                self.layers[layerIdx].next = self.layers[layerIdx + 1]
            else:
                self.layers[layerIdx].prev = self.layers[layerIdx - 1]
                self.layers[layerIdx].next = self.loss
                self.outputLayerActivation = self.layers[layerIdx]
                
            if hasattr(self.layers[layerIdx], "weight"):
                self.trainableLayers.append(self.layers[layerIdx])
        
        if self.loss is not None:
            self.loss.keepTrainableLayers(self.trainableLayers)

    def train(self, X, y, *, numEpoch=1, batchSize=None, printEvery=1):

        self.lossHist   = []
        self.accHist    = []
        self.lrHist     = []
        self.epochHist  = []

        self.accuracy.init(y)

        for epoch in range(1, numEpoch+1):
            
            if batchSize is None:
                numStep = 1
            else:
                numStep = X.shape[1] // batchSize
                if numStep*batchSize < X.shape[1]:
                    numStep += 1

            self.loss.resetAccumLoss()
            self.accuracy.resetAccumAcc()

            for step in range(numStep):
                
                if batchSize is None:
                    batchX = X
                    batchy = y
                else:
                    batchX = X[:, step*batchSize:(step+1)*batchSize]
                    batchy = y[step*batchSize:(step+1)*batchSize]

                output = self.forward(batchX, training=True)
                
                outputLoss, regLoss = self.loss.calculateLoss(output, batchy, includeRegLoss=True)
                loss = outputLoss + regLoss

                prediction = self.outputLayerActivation.prediction(output)
                acc = self.accuracy.calculateAccuracy(prediction, batchy)

                self.backward(output, batchy)
                
                self.optimizer.updateLearningRate()
                for layer in self.trainableLayers:
                    self.optimizer.updateLayer(layer)
                self.optimizer.updateStep()

                if not step % printEvery or step == numStep - 1:
                    print(  "  " +
                            f"epoch: {epoch}, " +
                            f"step: {step}, " + 
                            f"acc: {acc:.3f}, " + 
                            f"loss: {loss:.3f}, " + 
                            f"outputLoss: {outputLoss:.3f}, " + 
                            f"regLoss: {regLoss:.3f}, " + 
                            f"lr: {self.optimizer.learningRate:.3f}"
                            )
                    
            epochOutputLoss, epochRegLoss = self.loss.calculateAccumLoss(includeRegLoss=True)
            epochLoss = epochOutputLoss + epochRegLoss
            epochAcc = self.accuracy.calculateAccumAcc()

            #print(  f"EPOCH: {epoch}, -->" +
            #        f"acc: {epochAcc:.3f}, " + 
            #        f"loss: {epochLoss:.3f}, " + 
            #        f"outputLoss: {epochOutputLoss:.3f}, " + 
            #        f"regLoss: {epochRegLoss:.3f}, " + 
            #        f"lr: {self.optimizer.learningRate:.3f}"
            #        )    

            self.lossHist.  append  (epochLoss                  )
            self.accHist.   append  (epochAcc                   )
            self.lrHist.    append  (self.optimizer.learningRate)
            self.epochHist. append  (epoch                      )    

    def validate(self, XVal, yVal, batchSize=None):

        if batchSize is None:
            numStep = 1
        else:
            numStep = XVal.shape[1] // batchSize
            if numStep*batchSize < XVal.shape[1]:
                numStep += 1

        self.loss.resetAccumLoss()
        self.accuracy.resetAccumAcc()

        for step in range(numStep):
                
            if batchSize is None:
                batchXVal = XVal
                batchyVal = yVal
            else:
                batchXVal = XVal[:, step*batchSize:(step+1)*batchSize]
                batchyVal = yVal[step*batchSize:(step+1)*batchSize]

            output = self.forward(batchXVal, training=False)
                
            outputLoss = self.loss.calculateLoss(output, batchyVal)
            loss = outputLoss

            prediction = self.outputLayerActivation.prediction(output)
            acc = self.accuracy.calculateAccuracy(prediction, batchyVal)

        valLoss = self.loss.calculateAccumLoss()
        valAcc = self.accuracy.calculateAccumAcc()

        print(  f"validation, " + 
                f"acc: {valAcc:.3f}, " + 
                f"loss: {valLoss:.3f}" )

    def forward(self, X, training):

        self.layerInput.forward(X, training)

        for layer in self.layers:

            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):

        if isinstance(self.layers[-1], actSoftmax) and isinstance(self.loss, lossCatCrossEnt):

            softmaxCategorical = SoftmaxCategorical()
            softmaxCategorical.backward(output, y)
            self.layers[-1].dinputs = softmaxCategorical.dinputs

            for layer in reversed(self.layers[0:-1]):

                layer.backward(layer.next.dinputs)

        else:
            self.loss.backward(output, y)

            for layer in reversed(self.layers):

                layer.backward(layer.next.dinputs)

    def plotEpoch(self):
        plt.figure()
        plt.plot(self.epochHist, self.lossHist)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()

        plt.figure()
        plt.plot(self.epochHist, self.accHist)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid()

        plt.figure()
        plt.plot(self.epochHist, self.lrHist)
        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.grid()

    def getParams(self):
        
        params = []

        for layer in self.trainableLayers:

            params.append(layer.getParams())

        return params
    
    def setParams(self, params):

        for paramSet, layer in zip(params, self.trainableLayers):
            layer.setParams(*paramSet)

    def saveParams(self, path):
        
        with open(path, "wb") as f:
            pickle.dump(self.getParams(), f)

    def loadParams(self, path):
        with open(path, "rb") as f:
            self.setParams(pickle.load(f))

    def saveModel(self, path):

        modelCopy = copy.deepcopy(self)

        modelCopy.loss.resetAccumLoss()
        modelCopy.accuracy.resetAccumAcc()
        
        modelCopy.layerInput.__dict__.pop("output", None)
        modelCopy.loss.__dict__.pop("dinputs", None)

        for layer in modelCopy.layers:
            for property in ["inputs", "output", "dweight", "dbias", "dinputs"]:
                layer.__dict__.pop(property, None)

        with open(path, "wb") as f:
            pickle.dump(modelCopy, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, X, *, batchSize=None):
        
        if batchSize is None:
            numStep = 1
        else:
            numStep = X.shape[1] // batchSize
            if numStep*batchSize < X.shape[1]:
                numStep += 1

        output = []

        for step in range(numStep):

            if batchSize is None:
                batchX = X
            else:
                batchX = X[:, step*batchSize:(step+1)*batchSize]

            batchOutput = self.forward(batchX, training=False)

            output.append(batchOutput)

        return np.hstack(output)