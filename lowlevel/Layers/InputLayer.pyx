from AbstractLayer import AbstractLayer

cdef class InputLayer(AbstractLayer):
    def __init__(self, int number_of_neurons):
        super().__init__(number_of_neurons)
