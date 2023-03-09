import numpy as np


def activation_function(x):
    return 1 / (1 + np.exp(-x))


def activation_function_derivative(x):
    return (1-activation_function(x)) * activation_function(x)
