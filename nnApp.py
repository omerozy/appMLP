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

# give input and take output
layer1.forward(X)
act1.forward(layer1.output)

layer2.forward(act1.output)
act2.forward(layer2.output)

# calculate loss of outputs for given results
lossFunc = loss()
loss = lossFunc.calculate(act2.output, y)

# 
print("Loss:", loss)
#output = act2.output
#print(output[[0, 2], 0:7])
plt.scatter(X[0], X[1])
#plt.show()