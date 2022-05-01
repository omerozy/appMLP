from nnObjs import *

# define input
X, y = spiral_data(samples=100, classes=3)
X = np.array(X).T
y = np.array(y).T

# define layer properties (weigths, biases, activation functions)
# hidden layers
layer1 = layerDense(3, 2)   
act1 = actReLu()

# output layer
layer2 = layerDense(3, 3)
act2 = actSoftmax()

for i in range(1700):
    # give input and take output
    layer1.forward(X)
    act1.forward(layer1.output)

    layer2.forward(act1.output)
    act2.forward(layer2.output)

    # calculate loss of outputs for given results
    lossFunc = lossCatCrossEnt()
    loss = lossFunc.calculate(act2.output, y)

    # backpropogate
    lossFunc.backward(act2.output, y)
    dvalues1 = lossFunc.dinputs

    act2.backward(dvalues1)
    dvalues2 = act2.dinputs

    layer2.backward(dvalues2)
    dweightslayer2 = layer2.dweights
    dbiaseslayer2 = layer2.dbiases
    dvalues3 = layer2.dinputs

    act1.backward(dvalues3)
    dvalues4 = act1.dinputs

    layer1.backward(dvalues4)
    dweightslayer1 = layer1.dweights
    dbiaseslayer1 = layer1.dbiases

    # update weights and biases
    layer1.weights = np.subtract(layer1.weights, layer1.dweights)
    layer1.biases = np.subtract(layer1.biases, layer1.biases)

    layer2.weights = np.subtract(layer2.weights, layer2.dweights)
    layer2.biases = np.subtract(layer2.biases, layer2.biases)

    # give input and take output
    layer1.forward(X)
    act1.forward(layer1.output)

    layer2.forward(act1.output)
    act2.forward(layer2.output)

    # calculate loss of outputs for given results
    lossFunc = lossCatCrossEnt()
    loss = lossFunc.calculate(act2.output, y)
    print(str(loss) + "     " + str(i))
#######################################################
print("POST-PROCESS") 
print("Loss:", loss)

print(y[0:5])
print(act2.output[:, 0:5])

print(dvalues1[:, 0:5])
print(dvalues2[:, 0:5])
print(dweightslayer2[:, 0:5])
print(dbiaseslayer2[:, 0:5])
print(dvalues3[:, 0:5])
print(dvalues4[:, 0:5])
print(dweightslayer1[:, 0:5])
print(dbiaseslayer1[:, 0:5])

plt.scatter(X[0], X[1])
#plt.show()