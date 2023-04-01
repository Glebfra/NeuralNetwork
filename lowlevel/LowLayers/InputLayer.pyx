cimport numpy as np
from AbstractLayer cimport AbstractLayer


cdef class InputLayer(AbstractLayer):
    def __init__(self, int number_of_neurons):
        super().__init__(number_of_neurons)
        self.inputs = None
        self.outputs = None

    cpdef np.ndarray feed_forward(self, np.ndarray inputs):
        self.inputs = inputs.reshape((len(inputs), 1))
        self.outputs = self.inputs
        return self.outputs
