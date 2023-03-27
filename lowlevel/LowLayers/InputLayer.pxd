cimport numpy as np
from AbstractLayer cimport AbstractLayer

cdef class InputLayer(AbstractLayer):
    cpdef np.ndarray feed_forward(self, np.ndarray inputs)
