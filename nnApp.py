from nnObjs import *

class network:
    def __init__(self, layerList, actList):
        self.layerList = layerList.copy()
        self.actList = actList.copy()
        self.numLayer = len(self.layerList)
    def forward(self, X):
        for layerIdx in range(self.numLayer):
            if layerIdx == 0:
                self.layerList[layerIdx].forward(X)
                self.actList[layerIdx].forward(self.layerList[layerIdx].output)
            elif layerIdx != range(self.numLayer)[-1]:
                self.layerList[layerIdx].forward(self.actList[layerIdx - 1].output)
                self.actList[layerIdx].forward(self.layerList[layerIdx].output)
            elif layerIdx == range(self.numLayer)[-1]:
                self.layerList[layerIdx].forward(self.actList[layerIdx - 1].output)
                self.actList[layerIdx].forward(self.layerList[layerIdx].output)
                yPred = self.actList[layerIdx].output
                self.yPred = yPred
    def calculateLoss(self, yTrue):
            self.actList[-1].calculateLoss(yTrue)
            self.loss = self.actList[-1].lossValue
            acc = np.mean(np.argmax(self.yPred, axis = 0, keepdims=True) == yTrue)
            self.acc = acc
    def backward(self, yTrue):
        self.actList[-1].backward(self.actList[-1].output, yTrue)
        dvalues = self.actList[-1].dinputs
        self.layerList[-1].backward(dvalues)
        dvalues = self.layerList[-1].dinputs
        for layerIdx in reversed(range(0, self.numLayer - 1)):
            self.actList[layerIdx].backward(dvalues)
            dvalues = self.actList[layerIdx].dinputs
            self.layerList[layerIdx].backward(dvalues)
            dvalues = self.layerList[layerIdx].dinputs

# define input
X, y = spiral_data(samples=100, classes=3)
X = np.array(X).T
y = np.array(y).T

#
l = [64, 3]
numOptStep = 10000

#
layerList = []
actList = []
numLayers = len(l)
lossMin = 10
stepMin = 1
accMax = -1
optimizer = optimizerAdam(initialLearningRate=0.02, decay=1e-5)

# define layer properties (weigths, biases, activation functions)
for layerIdx in range(numLayers):   # first layer
    if layerIdx == 0:
        layer = layerDense(l[layerIdx], 2)   
        act = actReLu()

        layerList.append(layer)
        actList.append(act)
    elif layerIdx == range(numLayers)[-1]:  # output layer
        layer = layerDense(l[layerIdx], l[layerIdx - 1])   
        act = SoftmaxCategorical()

        layerList.append(layer)
        actList.append(act)
    else:                                   # other layers
        layer = layerDense(l[layerIdx], l[layerIdx - 1])   
        act = actReLu()

        layerList.append(layer)
        actList.append(act)

# optimization steps
for optStepIdx in range(numOptStep):
    NN = network(layerList, actList)
    NN.forward(X)
    NN.calculateLoss(y)

    if lossMin > NN.loss:
        lossMin = NN.loss
        stepMin = optStepIdx
        networkMinLoss = network(layerList.copy(), actList.copy())

    print(f"step: {optStepIdx}    " + f"acc: {NN.acc:.4f}    " + f"loss: {NN.loss:.4f}    " + f"lr: {optimizer.learningRate:.4f}")

    NN.backward(y)    

    # execute optimizer
    optimizer.updateLearningRate()
    for layerIdx in reversed(range(numLayers)):
        layer = layerList[layerIdx]
        optimizer.updateLayer(layer)
    optimizer.updateStep()

#######################################################
print(" ")
print("POST-PROCESS")
networkMinLoss.forward(X)
networkMinLoss.calculateLoss(y)
print("Min. Loss: " + str(networkMinLoss.loss) + " found in step " + str(stepMin) + " with accuracy " + str(networkMinLoss.acc))

#print(y[0:5])
#print(actList[-1].output[:, 0:5])

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
print(z.shape)    
plt.imshow(z, extent=[x1Range.min(), x1Range.max(), x2Range.min(), x2Range.max()], alpha=0.5, interpolation="nearest")

plt.show()