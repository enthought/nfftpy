"""
Setup extension modules for cython wrapper for NFFT.
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

INCLUDEPATH = '/usr/local/include'
LIBPATH = '/usr/local/lib'

nfft_params = dict(
    include_dirs=[INCLUDEPATH, numpy.get_include()],
    library_dirs=[LIBPATH],
    libraries=['nfft3'],
    # compile arguments taken from NFFT bash script 'simple_test'
    extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
                       '-fstrict-aliasing -ffast-math'.split(),
    extra_link_args=['-Wl,-rpath='+LIBPATH]
    )

def extension(name):
    return Extension(name=name, sources=[name+'.pyx'], **nfft_params)

ext_modules = [extension(name) for name in [
    'test',
    #'simple_test',
    #'simple_test_np'
    ]]

setup(
  name = 'NFFT wrapper for Python',
  ext_modules= ext_modules,
  cmdclass = {'build_ext': build_ext}
)


