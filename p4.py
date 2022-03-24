from audioop import bias
import numpy as np

inputs =    [   [1,     2,      3,      2.5 ],
                [2,     5,      -1,     2   ],
                [-1.5,  2.7,    3.3,    -0.8] ] # three outputs from three neurons in the previous layer
 
weights = [ [0.2,   0.8,    -0.5,   1   ],          # weights of three inputs
            [0.5,   -0.91,  0.26,   -0.5],     # weights of three inputs
            [-0.26, -0.27,  0.17,   0.87]]   # weights of three inputs

biases = [[2, 3, 0.5]]

weights2 = [ [  0.1,   -0.14,   0.5     ],          # weights of three inputs
            [   -0.5,  0.12,    -0.33   ],     # weights of three inputs
            [   -0.44, 0.73,    -0.13   ]]   # weights of three inputs

biases2 = [[-1, 2, -0.5]]

layer1Output = np.dot(weights, np.array(inputs).T) + np.array(biases).T
layer2Output = np.dot(weights2, layer1Output) + np.array(biases2).T
print(layer2Output)
