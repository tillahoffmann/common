from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = 'cython_helper',
    ext_modules = cythonize("cython_helper.pyx",
                            annotate=True),
)