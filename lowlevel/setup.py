from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        module_list=[
            'Layers/*.pyx',
            '*.pyx'
        ],
        build_dir="build"
    ),
    include_dirs=[numpy.get_include()],
)
