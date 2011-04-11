"""
Cython mapping of nfft3.h

First cut: only symbols from the first part of nfft3.h (used in simple_test)

"""

ctypedef complex fftw_complex

cdef extern from "nfft3.h":


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

        # Flags for precomputation, (de)allocation, and FFTW usage.
        # default is PRE_PHI_HUT| PRE_PSI| MALLOC_X| MALLOC_F_HAT| MALLOC_F|
        # FFTW_INIT| FFT_OUT_OF_PLACE
        unsigned nfft_flags

        # Nodes in time/spatial domain. Num elements = M_total * d
        double* x

    ctypedef enum:
        # Constant (bit mask) symbols. These are just placeholders with
        # dummy values. The actual values will come from nfft3.h
        FFTW_DESTROY_INPUT = 1
        FFTW_ESTIMATE = 2
        FFTW_INIT = 3
        FFT_OUT_OF_PLACE = 4
        FG_PSI = 5
        MALLOC_F = 6
        MALLOC_F_HAT = 7
        MALLOC_X = 8
        PRE_FG_PSI = 9
        PRE_FULL_PSI = 10
        PRE_LIN_PSI = 11
        PRE_ONE_PSI = 12
        PRE_PHI_HUT = 13
        PRE_PSI = 14

    # ==============================
    # Functions in alphabetic order
    # ==============================

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

    # Destroys a transform plan
    void nfft_finalize(nfft_plan* ths)

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

    # Precomputation for a transform plan.
    # if PRE_*_PSI is set the application program has to call this routine
    # (after) setting the nodes x
    void nfft_precompute_one_psi(nfft_plan* ths)

    # Computes a NFFT
    void nfft_trafo(nfft_plan* ths)
    void nfft_trafo_1d(nfft_plan* ths)
    void nfft_trafo_2d(nfft_plan* ths)
    void nfft_trafo_3d(nfft_plan* ths)
