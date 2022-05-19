from nnObjs import *

class network:

    def __init__(self, layerList, actList, dropoutList):
        self.layerList  = layerList
        self.actList    = actList
        self.numLayer   = len(self.layerList)
        self.dropoutList = dropoutList
    def forward(self, X):
        for layerIdx in range(self.numLayer):
            layer       = self.layerList[layerIdx   ]
            act         = self.actList  [layerIdx   ]
            actPrevious = self.actList  [layerIdx-1 ]

            if layerIdx == 0:
                layer.forward(X)
                act.forward(layer.output)
            elif layerIdx != range(self.numLayer)[-1]:
                layer.forward(actPrevious.output)
                act.forward(layer.output)
            elif layerIdx == range(self.numLayer)[-1]:
                layer.forward(actPrevious.output)
                act.forward(layer.output)
                yPred = act.output
                self.yPred = yPred

    def calculateLoss(self, yTrue):
        act = self.actList[-1]

        act.calculateOutputLoss(yTrue)
        self.loss = act.outputLoss
        for layerIdx in range(self.numLayer):
            layer = self.layerList[layerIdx]
            self.loss += act.calculateRegLoss(layer)
        self.acc = np.mean(np.argmax(self.yPred, axis = 0, keepdims=True) == yTrue)

    def backward(self, yTrue):
        layer = self.layerList[-1]
        act = self.actList[-1]

        act.backward(act.output, yTrue)
        dvalues = act.dinputs
        layer.backward(dvalues)
        dvalues = layer.dinputs

        for layerIdx in reversed(range(0, self.numLayer - 1)):
            layer = self.layerList[layerIdx]
            act = self.actList[layerIdx]

            act.backward(dvalues)
            dvalues = act.dinputs
            layer.backward(dvalues)
            dvalues = layer.dinputs

# define parameters
X, y = spiral_data(samples=1000, classes=3)
X = np.array(X).T
y = np.array(y).T
optimizer = optimizerAdam(initialLearningRate=0.02, decay=1e-5)
l = [512, 3]
dropoutIdxList = [0]
dropoutRate = 0.1
numOptStep = 10000

#
layerList = []
actList = []
dropoutList = {}
numLayers = len(l)
lossMin = 10
stepMin = 1
accMax = -1

# define layer properties (weigths, biases, activation functions)
for layerIdx in range(numLayers):   # first layer
    if layerIdx == 0:
        layer = layerDense(l[layerIdx], 2, weightRegularizerL2=5e-4, biasRegularizerL2=5e-4)   
        act = actReLu()

        layerList.append(layer)
        actList.append(act)
    elif layerIdx == range(numLayers)[-1]:  # output layer
        layer = layerDense(l[layerIdx], l[layerIdx - 1])   
        act = SoftmaxCategorical()

        layerList.append(layer)
        actList.append(act)
    else:                                   # other layers
        layer = layerDense(l[layerIdx], l[layerIdx - 1], weightRegularizerL2=5e-4, biasRegularizerL2=5e-4)   
        act = actReLu()

        layerList.append(layer)
        actList.append(act)
    
    if np.any(layerIdx == dropoutIdxList):
        dropoutList[layerIdx] = layerDropout(dropoutRate)

# optimization steps
for optStepIdx in range(numOptStep):
    NN = network(layerList, actList, dropoutList)
    NN.forward(X)
    NN.calculateLoss(y)

    if lossMin > NN.loss:
        lossMin = NN.loss
        stepMin = optStepIdx
        networkMinLoss = network(copy.deepcopy(layerList), copy.deepcopy(actList), dropoutList)

    print(f"step: {optStepIdx}    " + f"acc: {NN.acc:.4f}    " + f"loss: {NN.loss:.4f}    " + f"lr: {optimizer.learningRate:.4f}")

    NN.backward(y)    

    # execute optimizer
    optimizer.updateLearningRate()
    for layerIdx in reversed(range(numLayers)):
        layer = layerList[layerIdx]
        optimizer.updateLayer(layer)
    optimizer.updateStep()

#######################################################
###############    POST-PROCESS    ####################
#######################################################
print(" ")
print("POST-PROCESS")
networkMinLoss.forward(X)
networkMinLoss.calculateLoss(y)
print("Min. Loss: " + str(networkMinLoss.loss) + " found in step " + str(stepMin) + " with accuracy " + str(networkMinLoss.acc))

# validation
XVal, yVal = spiral_data(samples=np.shape(X)[1], classes=3)
XVal = np.array(XVal).T
yVal = np.array(yVal).T
networkMinLoss.forward(XVal)
networkMinLoss.calculateLoss(yVal)
print("validation, " + f"acc: {networkMinLoss.acc:.4f}    " + f"loss: {networkMinLoss.loss:.4f}")

#
networkMinLoss.forward(X)
networkMinLoss.calculateLoss(y)

redCode = np.array([255, 0, 0])/255
greenCode = np.array([0, 128, 0])/255
blueCode = np.array([0, 0, 255])/255

plt.figure(1)
yPred = np.argmax(networkMinLoss.yPred, axis=0, keepdims=True)
for pointIdx in range(X.shape[1]):
    if  yPred[0][pointIdx] == 0:
        plt.scatter(X[0][pointIdx], X[1][pointIdx], color=blueCode)
    elif  yPred[0][pointIdx] == 1:
        plt.scatter(X[0][pointIdx], X[1][pointIdx], color=greenCode)
    elif  yPred[0][pointIdx] == 2:
        plt.scatter(X[0][pointIdx], X[1][pointIdx], color=redCode)
plt.title("Estimation")

plt.figure(2)
for pointIdx in range(X.shape[1]):
    if  y[pointIdx] == 0:
        plt.scatter(X[0][pointIdx], X[1][pointIdx], color=blueCode)
    elif  y[pointIdx] == 1:
        plt.scatter(X[0][pointIdx], X[1][pointIdx], color=greenCode)
    elif  y[pointIdx] == 2:
        plt.scatter(X[0][pointIdx], X[1][pointIdx], color=redCode)
plt.title("Reference Data")

x1Range = np.arange(-1.5, 1.5, 10e-3)
x2Range = np.arange(-1.5, 1.5, 10e-3)
z = np.ones([len(x1Range), len(x2Range), 3])
x2Idx = 0
for x2 in reversed(x2Range):
    x1Idx = 0
    for x1 in x1Range:
        xSample = np.array([[x1, x2]]).T
        networkMinLoss.forward(xSample)
        yPred = networkMinLoss.yPred
        colorCode = blueCode*yPred[0] + greenCode*yPred[1] + redCode*yPred[2]
        z[x2Idx, x1Idx, :] = colorCode
        x1Idx += 1
    x2Idx += 1    
plt.imshow(z, extent=[x1Range.min(), x1Range.max(), x2Range.min(), x2Range.max()], alpha=0.5, interpolation="nearest")

plt.show()