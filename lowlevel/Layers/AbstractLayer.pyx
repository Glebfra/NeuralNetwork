import numpy as np
cimport numpy as np


cdef class AbstractLayer:
    cdef public np.ndarray weights
    cdef public np.ndarray biases

    cdef public AbstractLayer previous_layer
    cdef public int number_of_neurons

    def __init__(self, int number_of_neurons, AbstractLayer previous_layer=None):
        self.number_of_neurons = number_of_neurons
        self.previous_layer = previous_layer
        if self.previous_layer is not None:
            self.weights = np.random.random((self.number_of_neurons, self.previous_layer.number_of_neurons))
            self.biases = np.random.random((self.number_of_neurons, 1))
        else:
            self.weights = None
            self.biases = None

    cpdef np.ndarray activation_function(self, np.ndarray x):
        return 1/(1+np.exp(-x))

    cpdef np.ndarray feed_forward(self, np.ndarray inputs):
        if self.previous_layer is not None:
            return self.activation_function(self.weights @ inputs + self.biases)
        else:
            return inputs

    cpdef void back_propagation(self):
        pass
