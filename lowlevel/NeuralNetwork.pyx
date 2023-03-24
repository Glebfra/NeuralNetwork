import numpy as np
cimport numpy as np

cdef class NeuralNetwork:
    def __init__(self, int input_neurons, int hidden_neurons, int hidden_layers, int output_neurons):
        pass

    @classmethod
    def create_default_network(cls):
        pass

    cpdef add_input_layer(self, int number_of_neurons):
        pass

    cpdef add_hidden_layer(self, int number_of_neurons):
        pass

    cpdef add_hidden_layers(self, int number_of_neurons, int number_of_hidden_layers):
        pass

    cpdef add_output_layer(self, int number_of_neurons):
        pass

    cpdef feed_forward(self, np.ndarray inputs):
        pass

    cpdef back_propagation(self, np.ndarray train_inputs, np.ndarray train_outputs, int epochs, float learning_rate):
        pass
