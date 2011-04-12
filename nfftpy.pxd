"""
Interface module for Cython wrapper for NFFT libraries.
"""
from cnfft3 cimport fftw_complex, nfft_plan

cdef fftw_complex_array_to_numpy(fftw_complex *pca, int n)

cdef class NfftPlanWrapper:
    cdef nfft_plan plan

