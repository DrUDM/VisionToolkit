# -*- coding: utf-8 -*-
from distutils.core import setup

import setuptools
from Cython.Build import cythonize

setup(ext_modules=cythonize("c_alignment_algorithms.pyx"))
