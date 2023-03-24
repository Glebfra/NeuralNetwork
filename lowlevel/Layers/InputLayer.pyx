import numpy as np

from AbstractLayer import AbstractLayer

cdef class InputLayer(AbstractLayer):
    def __init__(self, int number_of_neurons):
        super().__init__(number_of_neurons)

    cpdef np.ndarray feed_forward(self, np.ndarray inputs):
        self.inputs = inputs.reshape((self.number_of_neurons, 1))
        self.outputs = self.inputs
        return self.inputs
