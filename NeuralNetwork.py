from Neuron import Neuron

import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_neurons_number, hidden_neurons_number, output_neurons_number):
        self.input_neurons_number = input_neurons_number
        self.hidden_neurons_number = hidden_neurons_number
        self.output_neurons_number = output_neurons_number

        self.hidden_neurons = [Neuron(weights=np.random.random(input_neurons_number), bias=0) for _ in range(hidden_neurons_number)]
        self.output_neurons = [Neuron(weights=np.random.random(hidden_neurons_number), bias=0) for _ in range(output_neurons_number)]

    def feed_forward(self, inputs: np.ndarray):
        if len(inputs) is not self.input_neurons_number:
            raise Exception('The inputs array cannot be incompatible with input neurons number')

        new_inputs = []
        for neuron in self.hidden_neurons:
            new_inputs.append(neuron.feed_forward(inputs))
        new_inputs = np.array(new_inputs)

        response = []
        for neuron in self.output_neurons:
            response.append(neuron.feed_forward(new_inputs))

        return response

    def back_propagation(self):
        pass

    def __str__(self):
        response = 'Hidden Neurons: \n'
        iteration = 1
        for neuron in self.hidden_neurons:
            response += f'[{iteration}] {str(neuron)} \n'

        response += 'Output Neurons: \n'
        iteration = 1
        for neuron in self.output_neurons:
            response += f'[{iteration}] {str(neuron)} \n'

        return response


if __name__ == '__main__':
    network = NeuralNetwork(3, 3, 1)
    inputs = np.array([1, 1, 0])
    print(network.feed_forward(inputs))
