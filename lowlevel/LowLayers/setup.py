from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    package_dir={
        'lowlevel.LowLayers': '',
    },
    ext_modules=cythonize(
        module_list=[
            '*.pyx'
        ],
    ),
    include_dirs=[numpy.get_include()],
)
