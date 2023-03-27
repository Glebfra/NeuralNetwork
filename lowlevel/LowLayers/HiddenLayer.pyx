import numpy as np


cdef class HiddenLayer(AbstractLayer):
    def __init__(self, int number_of_neurons, AbstractLayer previous_layer):
        super().__init__(number_of_neurons)
        self.previous_layer = previous_layer
        self.weights = np.random.random((self.number_of_neurons, self.previous_layer.number_of_neurons))
        self.biases = np.random.random((self.number_of_neurons, 1))

    cpdef np.ndarray back_propagation(self, np.ndarray error, float learning_rate):
        error = error.reshape((len(error), 1))
        out_error = self.weights.T @ error
        self.weights -= learning_rate * (error @ self.inputs.reshape((1, len(self.inputs))))
        self.biases -= learning_rate * error
        return out_error
