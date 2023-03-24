import numpy as np
cimport numpy as np
from cpython cimport array
from Layers.AbstractLayer cimport AbstractLayer

cdef class NeuralNetwork:
    cdef AbstractLayer input_layer
    cdef array.array hidden_layers

    cpdef add_input_layer(self, int number_of_neurons)
    cpdef add_hidden_layer(self, int number_of_neurons)
    cpdef add_hidden_layers(self, int number_of_neurons, int number_of_hidden_layers)
    cpdef add_output_layer(self, int number_of_neurons)
    cpdef feed_forward(self, np.ndarray inputs)
    cpdef back_propagation(self, np.ndarray train_inputs, np.ndarray train_outputs, int epochs, float learning_rate)
