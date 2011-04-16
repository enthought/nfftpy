"""
Interface module for Cython wrapper for NFFT libraries.

This is only needed to do cimports into other cython modules,
not for importing into pure python modules.

Fixme: Here we only declared those functions used in simple_test.
"""
cimport numpy as np

cimport cnfft3
from cnfft3 cimport fftw_complex, nfft_plan

cdef np.ndarray[np.complex128_t] fftw_complex_array_to_numpy(fftw_complex *pca,
                                                             int n_elems)
# Given a pointer to an array of fftw_complex, and its size,
# return a copy of the array as a complex numpy array.

cdef np.ndarray[np.double_t] float_array_to_numpy(double *pda, int n_elems)
# Given a pointer to an array of double, and its size,
# return a copy of the array as a double numpy array.


cdef fftw_complex_array_from_numpy(fftw_complex *pca, int n_elem,
                                   np.ndarray[np.complex128_t] arr)
# Copy numpy array to matching fftw complex C array


cdef float_array_from_numpy(double *pda, int n_elem,
                             np.ndarray[np.double_t] arr)
# Copy numpy array to matching double C array


cdef class NfftPlanWrapper:
    cdef nfft_plan plan
    cdef unsigned int _is_defined

