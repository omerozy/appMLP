from nnObjs import *

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
# define layer properties (weigths, biases, activation functions)
for layerIdx in range(numLayers):   # hidden layers
    if layerIdx == 0:
        layer = layerDense(l[layerIdx], 2)   
        act = actReLu()

        layerList.append(layer)
        actList.append(act)
    elif layerIdx == range(numLayers)[-1]:  # output layer
        layer = layerDense(l[layerIdx], l[layerIdx - 1])   
        act = actSoftmax()

        layerList.append(layer)
        actList.append(act)
    else:
        layer = layerDense(l[layerIdx], l[layerIdx - 1])   
        act = actReLu()

        layerList.append(layer)
        actList.append(act)

lossFunc = lossCatCrossEnt()
for optStepIdx in range(numOptStep):
    # forward
    for layerIdx in range(numLayers):
        if layerIdx == 0:
            layerList[layerIdx].forward(X)
            actList[layerIdx].forward(layerList[layerIdx].output)
        else:
            layerList[layerIdx].forward(actList[layerIdx - 1].output)
            actList[layerIdx].forward(layerList[layerIdx].output)
    loss = lossFunc.calculate(actList[-1].output, y)
    if lossMin > loss:
        stepMin = optStepIdx
        lossMin = loss 
        layerMin = layerList.copy()
        actMin = actList.copy()
    print(str(loss) + " " + str(optStepIdx))
    
    # backward
    lossFunc.backward(actList[-1].output, y)
    dvalues = lossFunc.dinputs
    for layerIdx in reversed(range(numLayers)):
        actList[layerIdx].backward(dvalues)
        dvalues = actList[layerIdx].dinputs
        layerList[layerIdx].backward(dvalues)
        dvalues = layerList[layerIdx].dinputs
        layerList[layerIdx].weights += -layerList[layerIdx].dweights
        layerList[layerIdx].biases += -layerList[layerIdx].dbiases

#######################################################
print(" ")
print("POST-PROCESS") 
print("Min. Loss: " + str(lossMin) + " found in step " + str(stepMin))

print(y[0:5])
print(actList[-1].output[:, 0:5])

redCode = np.array([255, 0, 0])/255
greenCode = np.array([0, 128, 0])/255
blueCode = np.array([0, 0, 255])/255

plt.figure(1)
yPred = np.argmax(actMin[-1].output, axis=0, keepdims=True)
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

def NN(X, layerList, actList):
    for layerIdx in range(len(layerList)):
            if layerIdx == 0:
                layerList[layerIdx].forward(X)
                actList[layerIdx].forward(layerList[layerIdx].output)
            else:
                layerList[layerIdx].forward(actList[layerIdx - 1].output)
                actList[layerIdx].forward(layerList[layerIdx].output)
    return actList[-1].output

x1Range = np.arange(-1.5, 1.5, 10e-3)
x2Range = np.arange(-1.5, 1.5, 10e-3)
z = np.ones([len(x1Range), len(x2Range), 3])
x1Idx = 0
for x1 in x1Range:
    x2Idx = 0
    for x2 in x2Range:
        xSample = np.array([[x1, x2]]).T
        yPred = NN(xSample, layerList, actList)
        colorCode = blueCode*yPred[0] + greenCode*yPred[1] + redCode*yPred[2]
        z[x1Idx, x2Idx, :] = colorCode
        x2Idx += 1
    x1Idx += 1
print(z.shape)    
plt.figure(2)
plt.imshow(z, extent=[x1Range.min(), x1Range.max(), x2Range.min(), x2Range.max()], alpha=0.5)

plt.show()