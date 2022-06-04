import os
import requests
import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from nnObjs import *

def loadMNISTDataset(path):

    labels = os.listdir(path)
    X = []
    y = []
    print("Loading images...")

    numFiles = 0
    for label in labels:
        files = os.listdir(path + "/" + label)
        numFiles += len(files)

    cumNum = 1
    for label in labels:
        files = os.listdir(path + "/" + label)
        for file in files:
            print(f"Loading file {file} in label {label} in {path}  PROGRESS: %{100*cumNum/numFiles:.3f}")
            image = cv2.imread(path + "/" + label + "/" + file, cv2.IMREAD_UNCHANGED)
            X.append(image)
            y.append(label)
            cumNum += 1

    return np.array(X), np.array(y).astype("uint8")

URL     = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
FILE    = 'fashion_mnist_images.zip'
FOLDER  = "fashionMNISTImages"


# prepare data
if not os.path.isfile(FILE):

    print(f'Downloading {URL} and saving as {FILE}...')
    
    r = requests.get(URL)
    with open(FILE, "wb") as zipFile:
        zipFile.write(r.content)

if not os.path.isdir(FOLDER):
    print(f'Unzipping to folder {FOLDER}...')

    with zipfile.ZipFile(FILE) as zipFile:
        zipFile.extractall(FOLDER)

# display example data
imageData = cv2.imread(FOLDER + "/train/4/0011.png", cv2.IMREAD_UNCHANGED)
np.set_printoptions(linewidth=200)
plt.    imshow  (imageData, cmap="gray")
plt.    show    (block=False)
plt.    pause   (3)
plt.    close   ()

#load train and test data
X,      y       = loadMNISTDataset  (   FOLDER     +   "/train"  )
XTest,  yTest   = loadMNISTDataset  (   FOLDER     +   "/test"   )

#scale image data (0, 255) to (-1, 1)
X       = (X    .astype(np.float32) -   255/2) /   (255/2)
XTest   = (XTest.astype(np.float32) -   255/2) /   (255/2)

#reshape
X       = X.        reshape(X.      shape[0], -1)
XTest   = XTest.    reshape(XTest.  shape[0], -1)
X       = np.transpose(X)
XTest   = np.transpose(XTest)

#shuffle
indexes = np.array(range(X.shape[1]))
np.random.shuffle(indexes)
X = X[:, indexes]
y = y[indexes]

# create model
model1 = model()

model1.addLayer(layerDense(64, X.shape[0]))
model1.addLayer(actReLu())
model1.addLayer(layerDense(64, 64))
model1.addLayer(actReLu())
model1.addLayer(layerDense(10, 64))
model1.addLayer(actSoftmax())

model1.set( loss        =   lossCatCrossEnt(), 
            optimizer   =   optimizerAdam(decay=5e-5),
            accuracy=accuracyCategorical()
            )

model1.establish()

model1.train(X, y, numEpoch=5, batchSize=128, printEvery=100)

model1.validate(XVal=XTest, yVal=yTest, batchSize=128)

model1.plotEpoch()

params = model1.getParams()

model1.saveParams("fashionMNIST.params")

model1.saveModel("fashionMNIST.model")

# CREATE NEW MODEL FROM TRAINED MODEL PARAMETERS
model2 = model()

model2.addLayer(layerDense(64, X.shape[0]))
model2.addLayer(actReLu())
model2.addLayer(layerDense(64, 64))
model2.addLayer(actReLu())
model2.addLayer(layerDense(10, 64)) 
model2.addLayer(actSoftmax())

model2.set(loss=lossCatCrossEnt(), accuracy=accuracyCategorical())

model2.establish()

model2.loadParams("fashionMNIST.params")

model2.validate(XTest, yTest)

# LOAD SAVED MODEL
model3 = model.load("fashionMNIST.model")

model3.validate(XTest, yTest)

confidences = model3.predict(XTest, batchSize=2500)
predictions = model3.outputLayerActivation.prediction(confidences)


# USE MODEL3
model3 = model.load("fashionMNIST.model")

fashionMNISTLabels =    {
                        0: 'T-shirt/top',
                        1: 'Trouser',
                        2: 'Pullover',
                        3: 'Dress',
                        4: 'Coat',
                        5: 'Sandal',
                        6: 'Shirt',
                        7: 'Sneaker',
                        8: 'Bag',
                        9: 'Ankle boot'
                        }

imageData = cv2.imread(FOLDER + "/test/4/0011.png", cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow  (imageData, cmap="gray")

#imageData = 255 - imageData
#plt.figure()
#plt.imshow  (imageData, cmap="gray")

imageData = cv2.resize(imageData, (28, 28))
plt.figure()
plt.imshow  (imageData, cmap="gray")

imageData  = (imageData    .astype(np.float32) -   255/2) /   (255/2)
imageData       = imageData.reshape(1, -1)
imageData       = np.transpose(imageData)


confidences = model3.predict(imageData)
predictions = model3.outputLayerActivation.prediction(confidences)

#print(predictions)
#print(predictions.shape)
#print(predictions[0, 0])

prediction = fashionMNISTLabels[predictions[0, 0]]

print(prediction)

plt.show()