cimport numpy as np
from AbstractLayer cimport AbstractLayer


cdef class HiddenLayer(AbstractLayer):
    cpdef np.ndarray back_propagation(self, np.ndarray error, float learning_rate)
