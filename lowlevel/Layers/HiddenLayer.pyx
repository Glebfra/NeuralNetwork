import numpy as np

from AbstractLayer import AbstractLayer


cdef class HiddenLayer(AbstractLayer):
    def __init__(self, int number_of_neurons, AbstractLayer previous_layer):
        super().__init__(number_of_neurons)
        self.previous_layer = previous_layer
        self.weights = np.random.random((self.number_of_neurons, self.previous_layer.number_of_neurons))
