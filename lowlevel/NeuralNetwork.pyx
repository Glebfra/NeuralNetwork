import numpy as np
cimport numpy as np

from LowLayers.InputLayer import InputLayer
from LowLayers.HiddenLayer import HiddenLayer
from LowLayers.OutputLayer import OutputLayer

cdef class NeuralNetwork:
    def __init__(self):
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None

    @classmethod
    def create_default_network(
            cls,
            number_of_input_neurons: int,
            number_of_hidden_neurons: int,
            number_of_hidden_layers: int,
            number_of_output_neurons: int
    ):
        class_object = cls()
        class_object.add_input_layer(number_of_input_neurons)
        class_object.add_hidden_layers(number_of_hidden_neurons, number_of_hidden_layers)
        class_object.add_output_layer(number_of_output_neurons)

        return class_object

    cpdef void add_input_layer(self, int number_of_neurons):
        self.input_layer = InputLayer(number_of_neurons=number_of_neurons)

    cpdef void add_hidden_layer(self, int number_of_neurons):
        if len(self.hidden_layers) == 0:
            self.hidden_layers.append(HiddenLayer(previous_layer=self.input_layer, number_of_neurons=number_of_neurons))
        self.hidden_layers.append(HiddenLayer(previous_layer=self.hidden_layers[len(self.hidden_layers)-1], number_of_neurons=number_of_neurons))

    cpdef void add_hidden_layers(self, int number_of_neurons, int number_of_hidden_layers):
        for layer in range(number_of_hidden_layers):
            self.add_hidden_layer(number_of_neurons)

    cpdef void add_output_layer(self, int number_of_neurons):
        self.output_layer = OutputLayer(previous_layer=self.hidden_layers[len(self.hidden_layers) - 1], number_of_neurons=number_of_neurons)

    cpdef np.ndarray feed_forward(self, np.ndarray inputs):
        inputs = inputs.reshape((self.input_layer.number_of_neurons, 1))
        output = self.input_layer.feed_forward(inputs)
        for hidden_layer in self.hidden_layers:
            output = hidden_layer.feed_forward(output)
        output = self.output_layer.feed_forward(output)
        return output

    cpdef float back_propagation(self, np.ndarray inputs, np.ndarray truth_out, float learning_rate):
        inputs = inputs.reshape((len(inputs), 1))
        truth_out = truth_out.reshape((len(truth_out), 1))

        out: np.ndarray = self.feed_forward(inputs)
        error: np.ndarray = self._error_derivative(out, truth_out)
        layer_error: np.ndarray = error * self.output_layer.get_derivative_error()
        out_error = self.output_layer.back_propagation(layer_error, learning_rate)
        for hidden_layer in reversed(self.hidden_layers):
            out_error = hidden_layer.back_propagation(out_error, learning_rate)

        return self._error(out, truth_out).mean()

    cpdef list train(self, np.ndarray train_inputs, np.ndarray train_outputs, int epochs, float learning_rate):
        error = []
        for epoch in range(epochs):
            input_error = []
            for index, train_input in enumerate(train_inputs):
                input_error.append(self.back_propagation(train_input, train_outputs[index], learning_rate))
            error.append(np.array(input_error).mean())
        return error

    cpdef np.ndarray _error(self, np.ndarray out, np.ndarray truth_out):
        return (out - truth_out) ** 2

    cpdef np.ndarray _error_derivative(self, np.ndarray out, np.ndarray truth_out):
        return 2 * (out - truth_out)
