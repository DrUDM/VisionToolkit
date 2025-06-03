# -*- coding: utf-8 -*-
import setuptools
from distutils.core import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize('c_comparison_algorithms.pyx'))

