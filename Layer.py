import numpy as np

from Neuron import Neuron
from functions import activation_function, activation_function_derivative


class Layer(object):
    def __init__(self, number_of_neurons, previous_layer):
        if number_of_neurons <= 0:
            raise ValueError('The neurons number cannot be zero or less than zero')
        self.number_of_neurons: int = number_of_neurons

        self.previous_layer: Layer | InputLayer = previous_layer

        self._weights: np.ndarray = np.random.random((self.number_of_neurons, self.previous_layer.number_of_neurons))
        self._biases: np.ndarray = np.random.random((self.number_of_neurons, 1))

        self.neurons = [Neuron(self._weights[i, :], self._biases[i]) for i in range(number_of_neurons)]

        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

        self._output: np.ndarray | None = None

    def feed_forward(self):
        self._output = self.activation_function(self._weights @ self.previous_layer.feed_forward() + self._biases)
        return self._output

    def set_activation_function(self, activation_function, activation_function_derivative):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    @property
    def output(self):
        return self._output

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
        for index, neuron in enumerate(self.neurons):
            neuron.weights = self._weights[index, :]

    def __str__(self):
        response = f'weights: {self._weights.tolist()} \n biases: {self._biases.tolist()} \n'
        return response


class InputLayer(object):
    def __init__(self, number_of_neurons):
        if number_of_neurons <= 0:
            raise ValueError('The neurons number cannot be zero or less than zero')
        self.number_of_neurons: int = number_of_neurons

        self._inputs: np.ndarray | None = None

        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def set_activation_function(self, activation_function, activation_function_derivative):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

    def feed_forward(self):
        if self._inputs is None:
            raise TypeError('Inputs cannot be None. Please set the inputs')
        return self._inputs.reshape((len(self._inputs), 1))

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: list | np.ndarray):
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        self._inputs = inputs


if __name__ == '__main__':
    input_layer = InputLayer(2)
    input_layer.inputs = [1, 2]
    hidden_layer = Layer(4, input_layer)
    output_layer = Layer(1, hidden_layer)
    print(output_layer.feed_forward())
