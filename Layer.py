import numpy as np

from Neuron import Neuron


class Layer(object):
    def __init__(self, number_of_neurons, previous_layer):
        if number_of_neurons <= 0:
            raise ValueError('The neurons number cannot be zero or less than zero')
        self.number_of_neurons: int = number_of_neurons

        self.previous_layer: Layer | InputLayer = previous_layer

        self._weights: np.ndarray = np.random.random((self.number_of_neurons, self.previous_layer.number_of_neurons))
        self._biases: np.ndarray = np.random.random((self.number_of_neurons, 1))

        self.neurons = [Neuron(self._weights[i, :], self._biases[i]) for i in range(number_of_neurons)]

        self.activation_function = lambda x: 1/(1+np.exp(-x))
        self.activation_function_derivative = lambda x: self.activation_function(x)*(1-self.activation_function(x))

        self.output: np.ndarray | None = None
        self.inputs: np.ndarray | None = None

    def feed_forward(self) -> np.ndarray:
        self.inputs = self.previous_layer.feed_forward()
        self.output = self.activation_function(self._weights @ self.inputs + self._biases)
        return self.output

    def back_propagation(self, error) -> np.ndarray:
        self.weights += error.reshape((len(error), 1)) @ self.inputs.reshape((1, len(self.inputs)))
        return self.weights.T @ error

    def set_activation_function(self, activation_function, activation_function_derivative):
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative

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
