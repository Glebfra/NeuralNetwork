from Layer import *


class NeuralNetwork(object):
    hidden_layers: list[Layer] | None
    input_layer: InputLayer | None
    output_layer: Layer | None

    number_of_input_neurons: int | None
    number_of_hidden_neurons: list[int] | None
    number_of_output_neurons: int | None

    def __init__(self):
        self.input_layer = None
        self.hidden_layers = None
        self.output_layer = None

        self.number_of_input_neurons = None
        self.number_of_hidden_neurons = None
        self.number_of_output_neurons = None

    def feed_forward(self, inputs):
        self.input_layer.inputs = inputs
        for hidden_layer in self.hidden_layers:
            hidden_layer.feed_forward()
        return self.output_layer.feed_forward()

    def add_input_layer(self, number_of_input_neurons):
        self.input_layer = InputLayer(number_of_input_neurons)
        self.number_of_input_neurons = number_of_input_neurons

    def add_hidden_layer(self, number_of_hidden_neurons):
        if self.hidden_layers is None:
            self.hidden_layers = [Layer(number_of_hidden_neurons, self.input_layer)]
        else:
            self.hidden_layers.append(Layer(number_of_hidden_neurons, self.hidden_layers[len(self.hidden_layers)-1]))

    def add_output_layer(self, number_of_output_neurons):
        self.output_layer = Layer(number_of_output_neurons, self.hidden_layers[len(self.hidden_layers)-1])

    def __str__(self):
        response = 'Hidden Layers \n'
        for index, hidden_layer in enumerate(self.hidden_layers):
            response += f'[{index}] {str(hidden_layer)}'

        response += 'Output Layer \n'
        response += str(self.output_layer)

        return response


if __name__ == '__main__':
    network = NeuralNetwork()
    network.add_input_layer(2)
    for _ in range(2):
        network.add_hidden_layer(5)
    network.add_output_layer(2)

    print(network.feed_forward(inputs=[0.5, 2]))
