import json

from Neuron import Neuron

import numpy as np
import matplotlib.pyplot as plt


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

    def back_propagation(self, training_inputs: np.ndarray, training_outputs: np.ndarray, epochs: int, training_speed=1.0):
        error = []
        errors = []
        for epoch in range(epochs):
            for index, training_input in enumerate(training_inputs):
                errors = []
                output = self.feed_forward(training_input)
                errors.append(self._error(training_outputs[index], output))

                for neuron_index, neuron in enumerate(self.output_neurons):
                    derivative_error = self._derivative_error(training_outputs[index][neuron_index], output[neuron_index])
                    weights_error = neuron.activation_function_derivative(neuron.inputs @ neuron.weights)
                    neuron.weights -= training_speed * derivative_error * weights_error * neuron.inputs.T

                for neuron_index, neuron in enumerate(self.hidden_neurons):
                    derivative_error = self._derivative_error(training_outputs[index][0], output[0])
                    weights_error = neuron.activation_function_derivative(neuron.inputs @ neuron.weights)
                    neuron.weights -= training_speed * derivative_error * weights_error * neuron.inputs.T
            error.append(np.array(errors).mean())
        return error

    def save_settings(self, filename):
        save = {
            'output_neurons': [],
            'hidden_neurons': []
        }

        with open(f'Settings/{filename}', 'w') as file:
            for index, neuron in enumerate(self.output_neurons):
                save['output_neurons'].append({
                    'id': index,
                    'weights': neuron.weights.tolist(),
                    'bias': neuron.bias
                })

            for index, neuron in enumerate(self.hidden_neurons):
                save['hidden_neurons'].append({
                    'id': index,
                    'weights': neuron.weights.tolist(),
                    'bias': neuron.bias
                })
            json.dump(save, file)

    def load_settings(self, filename):
        with open(f'Settings/{filename}', 'r') as file:
            settings = json.load(file)

        self.output_neurons = []
        for neuron in settings['output_neurons']:
            self.output_neurons.append(Neuron(np.array(neuron['weights']), float(neuron['bias'])))

        self.hidden_neurons = []
        for neuron in settings['hidden_neurons']:
            self.hidden_neurons.append(Neuron(np.array(neuron['weights']), float(neuron['bias'])))

    @staticmethod
    def _error(truth_out, out):
        return ((truth_out - out)**2).mean()

    @staticmethod
    def _derivative_error(truth_out, out):
        return -2*(truth_out - out)

    def __str__(self):
        response = 'Hidden Neurons: \n'
        iteration = 1
        for neuron in self.hidden_neurons:
            response += f'[{iteration}] {str(neuron)} \n'
            iteration += 1

        response += 'Output Neurons: \n'
        iteration = 1
        for neuron in self.output_neurons:
            response += f'[{iteration}] {str(neuron)} \n'
            iteration += 1

        return response


if __name__ == '__main__':
    network = NeuralNetwork(3, 1, 1)
    inputs_array = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    outputs_array = np.array([
        [0],
        [1],
        [0],
        [1],
        [0],
        [1],
        [0],
        [1]
    ])

    epochs = 500
    epochs_x = np.linspace(0, epochs, epochs)
    errors = network.back_propagation(inputs_array, outputs_array, epochs, 1)
    network.save_settings('test1.json')

    plt.plot(epochs_x, errors)
    plt.xlabel('epochs')
    plt.ylabel('error')
    plt.show()
