import numpy as np
cimport numpy as np

cdef class AbstractLayer:
    cpdef public np.ndarray weights
    cpdef public np.ndarray biases
    cpdef public np.ndarray inputs
    cpdef public np.ndarray outputs

    cpdef public int number_of_neurons

    cpdef AbstractLayer previous_layer;
    cpdef np.ndarray activation_function(self, np.ndarray x)
    cpdef np.ndarray activation_function_derivative(self, np.ndarray x)

    cpdef np.ndarray feed_forward(self, np.ndarray inputs)

    cpdef np.ndarray get_derivative_error(self)
