from matplotlib.pyplot import xlabel
from nnObjs import *
import sys

# CATERGORICAL CLASSIFIER
X, y = spiral_data(samples=1000, classes=3)
X = np.array(X).T
y = np.array(y).T
XVal, yVal = spiral_data(samples=100, classes=3)
XVal = np.array(XVal).T
yVal = np.array(yVal).T

model1 = model()

model1.addLayer(layerDense(512, 2, weightRegularizerL2=5e-4, biasRegularizerL2=5e-4))
model1.addLayer(actReLu())
model1.addLayer(layerDense(3, 512))
model1.addLayer(actSoftmax())

model1.set(     loss        =   lossCatCrossEnt(), 
                optimizer   =   optimizerAdam(initialLearningRate=0.02, decay=1e-5), 
                accuracy    =   accuracyCategorical())

model1.establish()

model1.train(X, y, numEpoch=1500, printEvery=100)

model1.validate(XVal, yVal)

### VISUALIZATION ###
redCode = np.array([255, 0, 0])/255
greenCode = np.array([0, 128, 0])/255
blueCode = np.array([0, 0, 255])/255
gridNum = 250
xMin = -1.5
xMax = 1.5
yMin = -1.5
yMax = 1.5

print("VISUALIZING RESULTS...")
plt.figure(1)
yPred = np.argmax(model1.forward(XVal, training=False), axis=0, keepdims=True)
for pointIdx in range(XVal.shape[1]):
    match  yPred[0][pointIdx]:
        case 0:
            colorCode = redCode
        case 1:
            colorCode = greenCode
        case 2:
            colorCode = blueCode
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
        case 2:
            colorCode = blueCode
    plt.scatter(XVal[0][pointIdx], XVal[1][pointIdx], color=colorCode)
plt.title("Reference Data")

XGrid = np.meshgrid(np.linspace(xMin, xMax, gridNum),np.linspace(yMin, yMax, gridNum))
xSample = np.empty([2, 1])
z = np.empty([gridNum, gridNum, 3])
for x1Idx in range(np.shape(XGrid)[1]):
    for x2Idx in range(np.shape(XGrid)[2]):
        xSample[0] = XGrid[0][x1Idx][x2Idx]
        xSample[1] = XGrid[1][x1Idx][x2Idx]
        yPred = model1.forward(xSample, training=False)
        colorCode = blueCode*yPred[0] + greenCode*yPred[1] + redCode*yPred[2]
        z[x1Idx][x2Idx] = colorCode

plt.figure(1)
plt.imshow(z, extent=[xMin, xMax, yMin, yMax], alpha=0.5, interpolation="nearest", origin="lower")

plt.figure(1)
plt.savefig("est.png")
plt.figure(2)
plt.savefig("ref.png")

plt.show()
