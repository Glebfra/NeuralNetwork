import numpy as np


class Neuron(object):
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def feed_forward(self, inputs):
        return self.weight * inputs + self.bias

    def back_propagation(self):
        pass
