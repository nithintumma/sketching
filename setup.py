from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(name='sketching', 
	 	ext_modules = cythonize('sketch.pyx'), 
	 	include_dirs = [np.get_include()])
