from matplotlib.pyplot import xlabel
from nnObjs import *
import sys

# REGRESSION
X, y = sine_data()
X = np.array(X).T
y = np.array(y).T

model3 = model()

model3.addLayer(layerDense(64, 1))
model3.addLayer(actReLu())
model3.addLayer(layerDense(64, 64))
model3.addLayer(actReLu())
model3.addLayer(layerDense(1, 64))
model3.addLayer(actLin())

model3.set(     loss        =   lossMeanSquaredError(), 
                optimizer   =   optimizerAdam(initialLearningRate=0.01, decay=1e-3), 
                accuracy    =   accuracyRegression())

model3.establish()

model3.train(X, y, numEpoch=3000, printEvery=100)

model3.plotEpoch()

plt.figure()
plt.plot(X[0, :], y[0, :])
yPred = model3.forward(X, training=False)
plt.plot(X[0, :], yPred[0, :])
plt.grid()
plt.show()