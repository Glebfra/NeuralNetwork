import numpy as np


cdef class AbstractLayer:
    def __init__(self, int number_of_neurons):
        self.inputs = None
        self.outputs = None
        self.previous_layer = None
        self.weights = None
        self.biases = None
        self.number_of_neurons = number_of_neurons

    cpdef np.ndarray activation_function(self, np.ndarray x):
        return 1/(1+np.exp(-x))

    cpdef np.ndarray activation_function_derivative(self, np.ndarray x):
        return self.activation_function(x) * (1-self.activation_function(x))

    cpdef np.ndarray feed_forward(self, np.ndarray inputs):
        self.inputs = inputs.reshape((len(inputs), 1))
        self.outputs = self.activation_function(self.weights @ self.inputs + self.biases)
        return self.outputs

    cpdef np.ndarray get_derivative_error(self):
        return self.activation_function_derivative(self.weights @ self.inputs + self.biases)
