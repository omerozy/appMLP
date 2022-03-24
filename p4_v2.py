import numpy as np

np.random.seed(0)

X =    [   [1,     2,      3,      2.5 ],
           [2,     5,      -1,     2   ],
           [-1.5,  2.7,    3.3,    -0.8] ]

X = np.array(X).T

class layerDense:
    def __init__(self, numNeurons, numInputs):
        self.weights = 0.1*np.random.randn(numNeurons, numInputs)
        self.biases = np.zeros((numNeurons, 1))
    def forward(self, inputs):
        self.output = np.dot(self.weights, inputs) + self.biases

layer1 = layerDense(5, 4)
layer2 = layerDense(2, 5)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)