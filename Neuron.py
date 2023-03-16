import numpy as np
from functions import activation_function, activation_function_derivative


class Neuron(object):
    def __init__(self, weights: np.ndarray, bias: float):
        self.weights: np.ndarray = weights.reshape((len(weights), 1))
        self.bias: float = bias
        self.inputs: np.ndarray = np.array([])
        self.output = None

        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def feed_forward(self, inputs: np.ndarray):
        inputs = inputs.reshape((1, len(inputs)))
        self.inputs = inputs
        self.output = float(self.activation_function(inputs @ self.weights + self.bias))
        return self.output

    def to_dict(self):
        return {'weights': self.weights.tolist(), 'bias': self.bias}

    def __str__(self):
        return f'weights: \n {self.weights} \n bias: {self.bias}'
