from matplotlib.pyplot import xlabel
from nnObjs import *
import sys

#### BINARY LOGISTIC REGRESSION
X, y = spiral_data(samples=100, classes=2)
X = np.array(X).T
y = np.array(y).T
#y.reshape(-1, 1)
XVal, yVal = spiral_data(samples=100, classes=2)
XVal = np.array(XVal).T
yVal = np.array(yVal).T
#yVal.reshape(-1, 1)

model2 = model()

model2.addLayer(layerDense(64, 2, weightRegularizerL2=5e-4, biasRegularizerL2=5e-4))
model2.addLayer(actReLu())
model2.addLayer(layerDense(1, 64))
model2.addLayer(actSigmoid())

model2.set(     loss        =   lossBinCrossEnt(), 
                optimizer   =   optimizerAdam(decay=5e-7), 
                accuracy    =   accuracyCategorical())

model2.establish()

model2.train(X, y, numEpoch=10000, printEvery=100)

model2.validate(XVal, yVal)

### VISUALIZATION ###
redCode = np.array([255, 0, 0])/255
greenCode = np.array([0, 128, 0])/255
#blueCode = np.array([0, 0, 255])/255
gridNum = 250
xMin = -1.5
xMax = 1.5
yMin = -1.5
yMax = 1.5

print("VISUALIZING RESULTS...")
plt.figure(1)
yPred = np.argmax(model2.forward(XVal, training=False), axis=0, keepdims=True)
for pointIdx in range(XVal.shape[1]):
    match  yPred[0][pointIdx]:
        case 0:
            colorCode = redCode
        case 1:
            colorCode = greenCode
    plt.scatter(XVal[0][pointIdx], XVal[1][pointIdx], color=colorCode)
plt.title("Estimation")

plt.figure(2)
plt.axis([xMin, xMax, yMin, yMax])
plt.axis("square")
for pointIdx in range(XVal.shape[1]):
    match  yVal[pointIdx]:
        case 0:
            colorCode = redCode
        case 1:
            colorCode = greenCode
    plt.scatter(XVal[0][pointIdx], XVal[1][pointIdx], color=colorCode)
plt.title("Reference Data")

XGrid = np.meshgrid(np.linspace(xMin, xMax, gridNum),np.linspace(yMin, yMax, gridNum))
xSample = np.empty([2, 1])
z = np.empty([gridNum, gridNum, 3])
for x1Idx in range(np.shape(XGrid)[1]):
    for x2Idx in range(np.shape(XGrid)[2]):
        xSample[0] = XGrid[0][x1Idx][x2Idx]
        xSample[1] = XGrid[1][x1Idx][x2Idx]
        yPred = model2.forward(xSample, training=False)
        if yPred[0]:
            colorCode = greenCode
        else:
            colorCode = redCode
        #colorCode = greenCode*yPred[0] + redCode*yPred[1]
        z[x1Idx][x2Idx] = colorCode

plt.figure(1)
plt.imshow(z, extent=[xMin, xMax, yMin, yMax], alpha=0.5, interpolation="nearest", origin="lower")

plt.figure(1)
plt.savefig("est.png")
plt.figure(2)
plt.savefig("ref.png")

plt.show()