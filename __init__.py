import numpy as np

activation_function = lambda x: 1 / (1 + np.exp(-x))
activation_function_derivative = lambda x: (1-activation_function(x))*activation_function(x)
