"""
Cython wrapper for NFFT libraries.

NFFT is a C subroutine library for computing the nonequispaced discrete
Fourier transform (NDFT) and its generalisations in one or more dimensions, of
arbitrary input size, and of complex data.
http://www-user.tu-chemnitz.de/~potts/nfft/

Work in progress

"""
import numpy as np
cimport numpy as np

from libc.string cimport memcpy

cimport cnfft3
from cnfft3 cimport fftw_complex
cimport cnfft3util

# ensure that our numpy complex is the same size as our NFFT/FFTW complex.
cdef int SIZEOF_COMPLEX = sizeof(np.complex128_t)
assert  SIZEOF_COMPLEX == sizeof(cnfft3.fftw_complex)


cdef fftw_complex_array_to_numpy(fftw_complex *pca, int n):
    """
    Given a pointer to an array of fftw_complex, and its size,
    return the array as a complex numpy array.
    For now, this just copies the data.
    """
    cdef np.ndarray[np.complex128_t] arr = np.empty(shape=n, dtype='complex128')
    memcpy(arr.data, pca, n * SIZEOF_COMPLEX)
    return arr


cdef class NfftPlanWrapper: #(cnfft3.NfftPlanHolder):
    """
    Thin class wrapper for NFFT functions which take an nfft_plan parameter
    """

    cdef cnfft3.nfft_plan plan

    # =======================================
    # Initialization and finalization methods
    # =======================================

    # Initialization class methods create and return a plan object
    # FIXME: Thought decorators not supported in cython but maybe they are?
    # FIXME: How to convert numpy arrays to int*?
    # http://mail.scipy.org/pipermail/numpy-discussion/2008-September/037675.html

    #def nfft_init(cls, int d, np.ndarray[np.int_t] N, int M):
        #cdef NfftPlanWrapper self
        #self = cls()
        #cnfft3.nfft_init(&(self.plan), &N, M)
        #return self
    #nfft_init = classmethod(nfft_init)

    def nfft_init_1d(cls, int N1, int M):
        cdef NfftPlanWrapper self
        self = cls()
        cnfft3.nfft_init_1d(&self.plan, N1, M)
        return self
    nfft_init_1d = classmethod(nfft_init_1d)

    def nfft_init_2d(cls, int N1, int N2, int M):
        cdef NfftPlanWrapper self
        self = cls()
        cnfft3.nfft_init_2d(&self.plan, N1, N2, M)
        return self
    nfft_init_2d = classmethod(nfft_init_2d)

    def nfft_init_3d(cls, int N1, int N2, int N3, int M):
        cdef NfftPlanWrapper self
        self = cls()
        cnfft3.nfft_init_3d(&self.plan, N1, N2, N3, M)
        return self
    nfft_init_3d = classmethod(nfft_init_3d)

    #def nfft_init_guru(cls, int d, np.ndarray[np.int_t] N, int M,
                        #np.ndarray[int] n, int m,
                        #unsigned nfft_flags, unsigned fftw_flags):
        #cdef NfftPlanWrapper self
        #self = cls()
        #cnfft3.nfft_init_guru(&self.plan, d, &N, M, &n, m,
                              #nfft_flags, fftw_flags)
        #return self
    #nfft_init_guru = classmethod(nfft_init_guru)

    # Finalization (before disposing of plan object)
    def nfft_finalize(self):
        cnfft3.nfft_finalize(&self.plan)

    # ==========================================================
    # Methods wrapping other NFFT functions, in alphabetic order
    # ==========================================================

    # Computes an adjoint NDFT
    def ndft_adjoint(self):
        cnfft3.ndft_adjoint(&self.plan)

    # Computes a NDFT
    def ndft_trafo(self):
        cnfft3.ndft_trafo(&self.plan)

    # Computes an adjoint NFFT
    def nfft_adjoint(self):
        cnfft3.nfft_adjoint(&self.plan)
    def nfft_adjoint_1d(self):
        cnfft3.nfft_adjoint_1d(&self.plan)
    def nfft_adjoint_2d(self):
        cnfft3.nfft_adjoint_2d(&self.plan)
    def nfft_adjoint_3d(self):
        cnfft3.nfft_adjoint_3d(&self.plan)

    # Checks a transform plan for frequently used bad parameter
    def nfft_check(self):
        cnfft3.nfft_check(&self.plan)

    # Precomputation for a transform plan.
    # if PRE_*_PSI is set the application program has to call this routine
    # (after) setting the nodes x
    def nfft_precompute_one_psi(self):
        cnfft3.nfft_precompute_one_psi(&self.plan)

    # Computes a NFFT
    def nfft_trafo(self):
        cnfft3.nfft_trafo(&self.plan)
    def nfft_trafo_1d(self):
        cnfft3.nfft_trafo_1d(&self.plan)
    def nfft_trafo_2d(self):
        cnfft3.nfft_trafo_2d(&self.plan)
    def nfft_trafo_3d(self):
        cnfft3.nfft_trafo_3d(&self.plan)


pw = NfftPlanWrapper.nfft_init_1d(10,10)
