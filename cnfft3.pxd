"""
Cython mapping of nfft3.h

First cut: only symbols from the first part of nfft3.h (used in simple_test)

"""
cimport numpy as np

cdef extern from "nfft3.h":

    # Type of complex data in FFTW. FFTW can optionally be compiled to complex.
    # We do not yet support this.
    ctypedef double fftw_complex[2]

    ctypedef struct nfft_plan:
        # for now, only declaring the few symbols used in simple_test:

        # Total number of samples (in f)
        int M_total

        # Total number of Fourier coefficients (in f_hat)
        int N_total

        # samples
        fftw_complex* f

        # Fourier coefficients
        fftw_complex* f_hat

        # dimension (rank)
        int d

        # multi-bandwidth (sizes of each dimension?)
        int *N

        # Flags for precomputation, (de)allocation, and FFTW usage.
        # default is PRE_PHI_HUT| PRE_PSI| MALLOC_X| MALLOC_F_HAT| MALLOC_F|
        # FFTW_INIT| FFT_OUT_OF_PLACE
        unsigned nfft_flags

        # Nodes in time/spatial domain. Num elements = M_total * d
        double* x

    ctypedef enum:
        # Names of constant (bit mask) symbols; values will come from nfft3.h
        FFTW_DESTROY_INPUT
        FFTW_ESTIMATE
        FFTW_INIT
        FFT_OUT_OF_PLACE
        FG_PSI
        MALLOC_F
        MALLOC_F_HAT
        MALLOC_X
        PRE_FG_PSI
        PRE_FULL_PSI
        PRE_LIN_PSI
        PRE_ONE_PSI
        PRE_PHI_HUT
        PRE_PSI


    # =========================================
    # Initialization and finalization functions
    # =========================================

    # Initialisation of a transform plan, simple:
        # ths The pointer to a nfft plan
        # d The dimension
        # N The multi bandwidth
        # M The number of nodes
    void nfft_init(nfft_plan* ths, int d, int* N, int M)

    # Initialisation of a transform plan, wrapper for d=1,2, or 3:
        # ths The pointer to a nfft plan
        # N1 bandwidth
        # N2 bandwidth
        # N3 bandwidth
        # M The number of nodes
    void nfft_init_1d(nfft_plan* ths, int N1, int M)
    void nfft_init_2d(nfft_plan* ths, int N1, int N2, int M)
    void nfft_init_3d(nfft_plan* ths, int N1, int N2, int N3, int M)

    # Initialisation of a transform plan, guru:
        # ths The pointer to a nfft plan
        # d The dimension
        # N The multi bandwidth
        # M The number of nodes
        # n The oversampled multi bandwidth
        # m The spatial cut-off
        # nfft_flags NFFT flags to use
        # fftw_flags_off FFTW flags to use
    void nfft_init_guru(nfft_plan* ths, int d, int* N, int M, int* n,
                        int m, unsigned nfft_flags, unsigned fftw_flags)

    # Destroys a transform plan
    void nfft_finalize(nfft_plan* ths)


    # ===================================
    # Other functions in alphabetic order
    # ===================================

    # Computes an adjoint NDFT
    void ndft_adjoint(nfft_plan* ths)

    # Computes a NDFT
    void ndft_trafo(nfft_plan* ths)

    # Computes an adjoint NFFT
    void nfft_adjoint(nfft_plan* ths)
    void nfft_adjoint_1d(nfft_plan* ths)
    void nfft_adjoint_2d(nfft_plan* ths)
    void nfft_adjoint_3d(nfft_plan* ths)

    # Checks a transform plan for frequently used bad parameter
    void nfft_check(nfft_plan* ths)

    # Precomputation for a transform plan.
    # if PRE_*_PSI is set the application program has to call this routine
    # (after) setting the nodes x
    void nfft_precompute_one_psi(nfft_plan* ths)

    # Computes a NFFT
    void nfft_trafo(nfft_plan* ths)
    void nfft_trafo_1d(nfft_plan* ths)
    void nfft_trafo_2d(nfft_plan* ths)
    void nfft_trafo_3d(nfft_plan* ths)

# =====================================================
# Numeric type definitions and sizes used in nfft_plan:
# =====================================================

ctypedef double nfft_float

# Sizes of numeric types used by nfft:
cdef enum:
    SIZEOF_INT = sizeof(int)
    SIZEOF_FLOAT = sizeof(nfft_float)
    SIZEOF_COMPLEX = sizeof(fftw_complex)

# Numpy cdef-usable dtypes that are size-compatible (though not always
# C-language assignment-compatible) with numeric types used by nfft:
ctypedef np.int32_t np_cdef_int
ctypedef np.float64_t np_cdef_float
ctypedef np.complex128_t np_cdef_complex
