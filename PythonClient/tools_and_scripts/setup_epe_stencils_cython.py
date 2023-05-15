from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='get_epe_stencils',
    ext_modules=cythonize("get_epe_stencils.pyx"),
    include_dirs=[numpy.get_include()]
)