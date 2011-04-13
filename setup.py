"""
Setup extension modules for cython wrapper for NFFT.

Command line parameters: build_ext --inplace, optionally followed by:
    --test testmod - to build from testmod.pyx
    otherwise builds from simple_test_class.pyx

"""
import sys

from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

INCLUDEPATH = '/usr/local/include'
LIBPATH = '/usr/local/lib'

nfft_params = dict(
    include_dirs=[INCLUDEPATH, numpy.get_include()],
    library_dirs=[LIBPATH],
    libraries=['nfft3'],
    # Compile arguments taken from NFFT bash script 'simple_test'.
    # To see preprocessor output as .i file, insert '-save-temps':
    extra_compile_args='-O3 -fomit-frame-pointer -malign-double '
                       '-fstrict-aliasing -ffast-math'.split(),
    extra_link_args=['-Wl,-rpath='+LIBPATH]
    )

def extension(name):
    return Extension(name=name, sources=[name+'.pyx'], **nfft_params)

def extmods(*names):
    return [extension(name) for name in names]

ext_modules = extmods('nfftpy')
if len(sys.argv) > 3:
    if sys.argv[-2] == '--test':
        ext_modules = extmods(sys.argv[-1])
        del sys.argv[-2:]

setup(
    name = 'NFFTPY',
    version = '0.1',
    description = "Cython wrapper for NFFT",
    author = 'Enthought, Inc.',
    author_email = 'info@enthought.com',
    url='https://github.com/enthought/nfftpy',
    license = 'BSD-like',

    cmdclass = {'build_ext': build_ext},
    ext_modules= ext_modules,

    install_requires = [
        "numpy >= 1.4.0",
        ],

    zip_safe = False,
    test_suite = 'nose.collector',
    tests_require = ['nose >= 0.9'],
)



