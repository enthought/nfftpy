"""
Setup extension modules for cython wrapper for NFFT.

Command line parameters: build_ext --inplace, optionally followed by:
    --temp - to build from temp.pyx
    --simple - to build from simple_test.pyx
    otherwise builds from simple_test_np.pyx

"""
import sys

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

ext_modules = extmods('simple_test_np')
if len(sys.argv) > 3:
    extra = sys.argv[-1]
    if extra == '--temp':
        ext_modules = extmods('temp')
        del sys.argv[-1]
    elif extra == '--simple':
        ext_modules = extmods('simple_test')
        del sys.argv[-1]

setup(
  name = 'NFFT wrapper for Python',
  ext_modules= ext_modules,
  cmdclass = {'build_ext': build_ext}
)


