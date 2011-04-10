"""
Cython wrapper for NFFT libraries.

NFFT is a C subroutine library for computing the nonequispaced discrete
Fourier transform (NDFT) and its generalisations in one or more dimensions, of
arbitrary input size, and of complex data.
http://www-user.tu-chemnitz.de/~potts/nfft/

Work in progress - not tested

"""
#import numpy as np
#cimport numpy as np

cimport cnfft3
import cnfft3
cimport cnfft3util

cdef class NfftPlanWrapper(cnfft3.NfftPlanHolder):
    """
    Thin class wrapper for NFFT functions which take an nfft_plan parameter
    """

    #cdef cnfft3.nfft_plan plan

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

    #def nfft_init_advanced(cls, int d, np.ndarray[np.int_t] N, int M,
                            #unsigned nfft_flags_on, unsigned nfft_flags_off):
        #cdef NfftPlanWrapper self
        #self = cls()
        #cnfft3.nfft_init_advanced(&self.plan, d, &N, M,
                                  #nfft_flags_on, nfft_flags_off)
        #return self
    #nfft_init_advanced = classmethod(nfft_init_advanced)

    #def nfft_init_guru(cls, int d, np.ndarray[np.int_t] N, int M,
                        #np.ndarray[int] n, int m,
                        #unsigned nfft_flags, unsigned fftw_flags):
        #cdef NfftPlanWrapper self
        #self = cls()
        #cnfft3.nfft_init_guru(&self.plan, d, &N, M, &n, m,
                              #nfft_flags, fftw_flags)
        #return self
    #nfft_init_guru = classmethod(nfft_init_guru)


    # Computations with plan object:

    def ndft_adjoint(self):
        cnfft3.ndft_adjoint(&self.plan)
    def ndft_trafo(self):
        cnfft3.ndft_trafo(&self.plan)
    def nfft_adjoint(self):
        cnfft3.nfft_adjoint(&self.plan)
    def nfft_adjoint_1d(self):
        cnfft3.nfft_adjoint_1d(&self.plan)
    def nfft_adjoint_2d(self):
        cnfft3.nfft_adjoint_2d(&self.plan)
    def nfft_adjoint_3d(self):
        cnfft3.nfft_adjoint_3d(&self.plan)
    def nfft_check(self):
        cnfft3.nfft_check(&self.plan)
    def nfft_precompute_full_psi(self):
        cnfft3.nfft_precompute_full_psi(&self.plan)
    def nfft_precompute_lin_psi(self):
        cnfft3.nfft_precompute_lin_psi(&self.plan)
    def nfft_precompute_one_psi(self):
        cnfft3.nfft_precompute_one_psi(&self.plan)
    def nfft_precompute_psi(self):
        cnfft3.nfft_precompute_psi(&self.plan)
    def nfft_trafo(self):
        cnfft3.nfft_trafo(&self.plan)
    def nfft_trafo_1d(self):
        cnfft3.nfft_trafo_1d(&self.plan)
    def nfft_trafo_2d(self):
        cnfft3.nfft_trafo_2d(&self.plan)
    def nfft_trafo_3d(self):
        cnfft3.nfft_trafo_3d(&self.plan)

    # Finalization (before disposing of plan object)
    def nfft_finalize(self):
        cnfft3.nfft_finalize(&self.plan)

