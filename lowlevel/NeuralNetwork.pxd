import numpy as np
cimport numpy as np
from LowLayers.AbstractLayer cimport AbstractLayer

cdef class NeuralNetwork:
    cdef AbstractLayer input_layer
    cdef list hidden_layers
    cdef AbstractLayer output_layer

    cpdef void add_input_layer(self, int number_of_neurons)
    cpdef void add_hidden_layer(self, int number_of_neurons)
    cpdef void add_hidden_layers(self, int number_of_neurons, int number_of_hidden_layers)
    cpdef void add_output_layer(self, int number_of_neurons)
    cpdef np.ndarray feed_forward(self, np.ndarray inputs)
    cpdef float back_propagation(self, np.ndarray inputs, np.ndarray truth_out, float learning_rate)
    cpdef list train(self, np.ndarray train_inputs, np.ndarray train_outputs, int epochs, float learning_rate)

    cpdef np.ndarray _error(self, np.ndarray out, np.ndarray truth_out)
    cpdef np.ndarray _error_derivative(self, np.ndarray out, np.ndarray truth_out)
