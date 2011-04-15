"""
Cython wrapper for NFFT libraries.

"NFFT is a C subroutine library for computing the nonequispaced discrete
Fourier transform (NDFT) and its generalisations in one or more dimensions, of
arbitrary input size, and of complex data."
http://www-user.tu-chemnitz.de/~potts/nfft/

Class NfftPlanWrapper is the heart of the module.

For example of using this module from python, see unit tests in
    ./tests/test_nfftpy.py
For example of using this module from another cython module, see
    simple_test_class.pyx

Work in progress

"""
import numpy as np
cimport numpy as np

from libc.string cimport memcpy

cimport cnfft3
from cnfft3 cimport fftw_complex
cimport cnfft3util

# bit mask constants for client modules to import:
FFTW_DESTROY_INPUT = cnfft3.FFTW_DESTROY_INPUT
FFTW_ESTIMATE = cnfft3.FFTW_ESTIMATE
FFTW_INIT = cnfft3.FFTW_INIT
FFT_OUT_OF_PLACE = cnfft3.FFT_OUT_OF_PLACE
FG_PSI = cnfft3.FG_PSI
MALLOC_F = cnfft3.MALLOC_F
MALLOC_F_HAT = cnfft3.MALLOC_F_HAT
MALLOC_X = cnfft3.MALLOC_X
PRE_FG_PSI = cnfft3.PRE_FG_PSI
PRE_FULL_PSI = cnfft3.PRE_FULL_PSI
PRE_LIN_PSI = cnfft3.PRE_LIN_PSI
PRE_ONE_PSI = cnfft3.PRE_ONE_PSI
PRE_PHI_HUT = cnfft3.PRE_PHI_HUT
PRE_PSI = cnfft3.PRE_PSI


cdef int SIZEOF_DOUBLE = sizeof(np.double_t)
cdef int SIZEOF_INT = sizeof(np.int_t)

# ensure that our numpy complex is the same size as our NFFT/FFTW complex.
cdef int SIZEOF_COMPLEX = sizeof(np.complex128_t)
assert  SIZEOF_COMPLEX == sizeof(cnfft3.fftw_complex)


# =============================================================================
# Conversion routines between C arrays & Numpy arrays (double & complex double)
# =============================================================================

# C arrays to numpy
# -----------------

cdef np.ndarray[np.complex128_t] fftw_complex_array_to_numpy(fftw_complex *pca,
                                                             int n_elems):
    """
    Given a pointer to an array of fftw_complex, and its size,
    return a copy of the array as a complex numpy array.
    """
    cdef np.ndarray[np.complex128_t] arr = np.empty(shape=n_elems,
                                                    dtype='complex128')
    memcpy(arr.data, pca, n_elems * SIZEOF_COMPLEX)
    return arr

cdef np.ndarray[np.double_t] double_array_to_numpy(double *pda, int n_elems):
    """
    Given a pointer to an array of double, and its size,
    return a copy of the array as a double numpy array.
    """
    cdef np.ndarray[np.double_t] arr = np.empty(shape=n_elems, dtype='double')
    memcpy(arr.data, pda, n_elems * SIZEOF_DOUBLE)
    return arr

cdef np.ndarray[np.int_t] int_array_to_numpy(int *pda, int n_elems):
    """
    Given a pointer to an array of int, and its size,
    return a copy of the array as an int numpy array.
    """
    cdef np.ndarray[np.int_t] arr = np.empty(shape=n_elems, dtype='int')
    memcpy(arr.data, pda, n_elems * SIZEOF_INT)
    return arr

# C arrays from numpy
# -------------------

cdef _array_from_numpy(void* ptr, int n_elems, int elem_size, arr,
                       void* parrdata):
    """
    Given a pointer to a C array, and its size and elment size,
    and a matching numpy array (c-ordered, contiguous),
    copy the numpy array into the C array.
    FIXME: don't require contiguity. Don't force copy.
    """
    if not arr.flags.c_contiguous:
        raise TypeError('input array must be C-contiguous')
    if arr.itemsize != elem_size:
        raise TypeError('input and output array elements must be the same size')
    if arr.shape != (n_elems,):
        raise TypeError('input and output arrays must have the same dimensions')
    memcpy(ptr, parrdata, n_elems * elem_size)

cdef fftw_complex_array_from_numpy(fftw_complex *pca, int n_elem,
                                   np.ndarray[np.complex128_t] arr):
    """ Copy numpy array to matching fftw complex C array
    """
    _array_from_numpy(pca, n_elem, SIZEOF_COMPLEX, arr, <void*>(arr.data))

cdef double_array_from_numpy(double *pda, int n_elem,
                             np.ndarray[np.double_t] arr):

    """ Copy numpy array to matching double C array
    """
    _array_from_numpy(pda, n_elem, SIZEOF_DOUBLE, arr, <void*>(arr.data))


cdef int_array_from_numpy(int *pda, int n_elem,
                             np.ndarray[np.int_t] arr):

    """ Copy numpy array to matching int C array
    """
    _array_from_numpy(pda, n_elem, SIZEOF_INT, arr, <void*>(arr.data))


# =============================================================
# Wrapper class for NFFT Plan - this is the heart of the module
# =============================================================


cdef class NfftPlanWrapper:
    """
    Thin class wrapper for NFFT functions which take an nfft_plan parameter.

    Instantiate with one of the class methods nfft_init*.
    See module tests/test_nfftpy.py for examples of usage.
    """

    def _init_(self):
        """
        Class is not intended to be instantiated directly. Instead, use
        the initialization class methods below.
        """
        self._is_defined = False

    def _check_defined(self):
        if not self._is_defined:
            raise RuntimeError('Attempted a method call on an undefined '
                               'nfft_plan')

    # ---------------------------------------
    # Initialization and finalization methods
    # ---------------------------------------

    # Initialization class methods create and return a plan object

    @classmethod
    def nfft_init(cls, int d, np.ndarray[np.int_t] N, int M):
        cdef NfftPlanWrapper self
        self = cls()
        cnfft3.nfft_init(&(self.plan), d, <int*>(N.data), M)
        self._is_defined = True
        return self

    @classmethod
    def nfft_init_1d(cls, int N1, int M):
        cdef NfftPlanWrapper self
        self = cls()
        cnfft3.nfft_init_1d(&self.plan, N1, M)
        self._is_defined = True
        return self

    @classmethod
    def nfft_init_2d(cls, int N1, int N2, int M):
        cdef NfftPlanWrapper self
        self = cls()
        cnfft3.nfft_init_2d(&self.plan, N1, N2, M)
        self._is_defined = True
        return self

    @classmethod
    def nfft_init_3d(cls, int N1, int N2, int N3, int M):
        cdef NfftPlanWrapper self
        self = cls()
        cnfft3.nfft_init_3d(&self.plan, N1, N2, N3, M)
        self._is_defined = True
        return self

    @classmethod
    def nfft_init_guru(cls, int d, np.ndarray[np.int_t] N, int M,
                        np.ndarray[int] n, int m,
                        unsigned nfft_flags, unsigned fftw_flags):
        cdef NfftPlanWrapper self
        self = cls()
        cnfft3.nfft_init_guru(&self.plan, d, <int*>(N.data),
                              M, <int*>(n.data), m,
                              nfft_flags, fftw_flags)
        self._is_defined = True
        return self

    # Finalization (before disposing of plan object)
    def nfft_finalize(self):
        self._check_defined()
        cnfft3.nfft_finalize(&self.plan)
        self._is_defined = False

    # ----------------------------------------------------------
    # Methods wrapping other NFFT functions, in alphabetic order
    # ----------------------------------------------------------

    # Computes an adjoint NDFT
    def ndft_adjoint(self):
        self._check_defined()
        cnfft3.ndft_adjoint(&self.plan)

    # Computes a NDFT
    def ndft_trafo(self):
        self._check_defined()
        cnfft3.ndft_trafo(&self.plan)

    # Computes an adjoint NFFT
    def nfft_adjoint(self):
        self._check_defined()
        cnfft3.nfft_adjoint(&self.plan)

    def nfft_adjoint_1d(self):
        self._check_defined()
        cnfft3.nfft_adjoint_1d(&self.plan)

    def nfft_adjoint_2d(self):
        self._check_defined()
        cnfft3.nfft_adjoint_2d(&self.plan)

    def nfft_adjoint_3d(self):
        self._check_defined()
        cnfft3.nfft_adjoint_3d(&self.plan)

    # Checks a transform plan for frequently used bad parameter
    def nfft_check(self):
        self._check_defined()
        cnfft3.nfft_check(&self.plan)

    # Precomputation for a transform plan.
    # if PRE_*_PSI is set the application program has to call this routine
    # (after) setting the nodes x
    def nfft_precompute_one_psi(self):
        self._check_defined()
        cnfft3.nfft_precompute_one_psi(&self.plan)

    # Computes a NFFT
    def nfft_trafo(self):
        self._check_defined()
        cnfft3.nfft_trafo(&self.plan)

    def nfft_trafo_1d(self):
        self._check_defined()
        cnfft3.nfft_trafo_1d(&self.plan)

    def nfft_trafo_2d(self):
        self._check_defined()
        cnfft3.nfft_trafo_2d(&self.plan)

    def nfft_trafo_3d(self):
        self._check_defined()
        cnfft3.nfft_trafo_3d(&self.plan)


    # ------------------------------------
    # Access to nfft_plan fields (members)
    # ------------------------------------
    # FIXME: consider giving these fields more informative names.
    # Now, we are keeping the original FFTW and NFFT names.
    # FIXME: when cython supports property decorator setters, use them.

    # -------------------------------------------------------------------------
    # Data sizes and flags are set through initialization methods, and then are
    # read-only.
    # FIXME: this is probably an unnecessary restriction, but for the first cut
    #        we are mimicking NFFT's simplest examples.

    # Total number of samples (in f)
    @property
    def M_total(self):
        self._check_defined()
        return self.plan.M_total

    # Total number of Fourier coefficients (in f_hat)
    @property
    def N_total(self):
        self._check_defined()
        return self.plan.N_total

    # dimension (rank)
    @property
    def d(self):
        self._check_defined()
        return self.plan.d

    # Flags for precomputation, (de)allocation, and FFTW usage.
    # default is PRE_PHI_HUT| PRE_PSI| MALLOC_X| MALLOC_F_HAT| MALLOC_F|
    # FFTW_INIT| FFT_OUT_OF_PLACE
    @property
    def nfft_flags(self):
        self._check_defined()
        return self.plan.nfft_flags

    # ----------------------------------------------------------------------
    # Array access is via copying the data into a new numpy array each time,
    # so client module should keep a reference to an array rather than reading
    # a property repeatedly.

    # samples in time domain (complex). Num elements = M_total
    f = property(_f_getter, _f_setter)

    def _f_getter(self):
        self._check_defined()
        cdef nfft_plan plan = self.plan
        return fftw_complex_array_to_numpy(plan.f, plan.M_total)

    def _f_setter(self, arr):
        self._check_defined()
        cdef nfft_plan plan = self.plan
        fftw_complex_array_from_numpy(plan.f, plan.M_total, arr)

    # Fourier coefficients (complex). Num elements = N_total
    f_hat = property(_f_hat_getter, _f_hat_setter)

    def _f_hat_getter(self):
        self._check_defined()
        cdef nfft_plan plan = self.plan
        return fftw_complex_array_to_numpy(plan.f_hat, plan.N_total)

    def _f_hat_setter(self, arr):
        self._check_defined()
        cdef nfft_plan plan = self.plan
        fftw_complex_array_from_numpy(plan.f_hat, plan.N_total, arr)

    # Nodes in time/spatial domain (double). Num elements = M_total * d
    x = property(_x_getter, _x_setter)

    def _x_getter(self):
        self._check_defined()
        cdef nfft_plan plan = self.plan
        nelem = plan.M_total * plan.d
        return double_array_to_numpy(plan.x, nelem)

    def _x_setter(self, arr):
        self._check_defined()
        cdef nfft_plan plan = self.plan
        nelem = plan.M_total * plan.d
        double_array_from_numpy(plan.x, nelem, arr)

    # "multi-bandwidth" (integer). Apparently num elements = d

    N = property(_N_getter, _N_setter)

    def _N_getter(self):
        self._check_defined()
        cdef nfft_plan plan = self.plan
        nelem = plan.d
        return int_array_to_numpy(plan.N, nelem)

    def _N_setter(self, arr):
        self._check_defined()
        cdef nfft_plan plan = self.plan
        nelem = plan.d
        int_array_from_numpy(plan.N, nelem, arr)

