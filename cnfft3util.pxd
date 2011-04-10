"""
Cython mapping of nfft3util.h

First cut: only those symbols used in simple_test

"""
from cnfft3 cimport fftw_complex

cdef extern from "nfft3util.h":
    # Inits a vector of random double numbers in [-0.5, 0.5]
    void nfft_vrand_shifted_unit_double(double *x, int n)

    # Inits a vector of random complex numbers in [0,1] x [0,1]
    void nfft_vrand_unit_complex(fftw_complex *x, int n)

    # Prints a vector of complex numbers (note cython needs 'const' removed.)
    void nfft_vpr_complex(fftw_complex *x, int n, char *text)


