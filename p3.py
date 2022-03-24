import numpy as np

inputs = [1, 2, 3, 2.5] # three outputs from three neurons in the previous layer

weights = [ [0.2, 0.8, -0.5, 1],          # weights of three inputs
            [0.5, -0.91, 0.26, -0.5],     # weights of three inputs
            [-0.26, -0.27, 0.17, 0.87]]   # weights of three inputs

biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases

print(output)