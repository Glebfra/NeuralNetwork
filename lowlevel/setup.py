from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('Layers/AbstractLayer.pyx'),
    include_dirs=[numpy.get_include()],
)
