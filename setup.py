from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "panoptic_halos.cython_funcs",
        sources=["panoptic_halos/cython_funcs.pyx"],
        include_dirs=[np.get_include()]  # Include NumPy's header files
    )
]

setup(
    name="panoptic_halos",
    author="Daniel Lopez",
    author_email="daniellopezcano13@gmail.com",
    version="0.1.0",
    ext_modules = cythonize(extensions),
    packages=["panoptic_halos"],
)