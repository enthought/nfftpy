"""
Direct cythonization of nfft-3.1.3/examples/nfft/simple_test.c
"""
import time

from cnfft3 cimport nfft_plan, nfft_init_1d, nfft_init_guru, \
    nfft_precompute_one_psi, \
    ndft_trafo, nfft_trafo, ndft_adjoint, nfft_adjoint, nfft_finalize, \
    PRE_PHI_HUT, FG_PSI, PRE_LIN_PSI, PRE_FG_PSI, PRE_PSI, PRE_FULL_PSI, \
    MALLOC_X, MALLOC_F_HAT, MALLOC_F, FFT_OUT_OF_PLACE, FFTW_INIT, PRE_ONE_PSI,\
    FFTW_ESTIMATE, FFTW_DESTROY_INPUT

from cnfft3util cimport nfft_vrand_shifted_unit_double, \
    nfft_vrand_unit_complex, nfft_vpr_complex

def nfft_second():
    "replacing nfft timer with python timer"
    return time.time()

def printf(fstring, *vals):
    "replacing C print with python print"
    print fstring % vals

def simple_test_nfft_1d():
    cdef nfft_plan p
    cdef double t
    cdef int N=14
    cdef int M=19
    cdef int n=32

    # init an one dimensional plan
    nfft_init_1d(&p, N, M)

    # init pseudo random nodes
    nfft_vrand_shifted_unit_double(p.x, p.M_total)

    # precompute psi, the entries of the matrix B
    if p.nfft_flags & PRE_ONE_PSI:
        nfft_precompute_one_psi(&p)
    # init pseudo random Fourier coefficients and show them
    nfft_vrand_unit_complex(p.f_hat, p.N_total)
    nfft_vpr_complex(p.f_hat, p.N_total,
        "given Fourier coefficients, vector f_hat")

    # direct trafo and show the result
    t=nfft_second()
    ndft_trafo(&p)
    t=nfft_second() - t
    nfft_vpr_complex(p.f, p.M_total, "ndft, vector f")
    printf(" took %e seconds.", t)

    # approx. trafo and show the result
    nfft_trafo(&p)
    nfft_vpr_complex(p.f, p.M_total, "nfft, vector f")

    # approx. adjoint and show the result
    ndft_adjoint(&p)
    nfft_vpr_complex(p.f_hat, p.N_total, "adjoint ndft, vector f_hat")

    # approx. adjoint and show the result
    nfft_adjoint(&p)
    nfft_vpr_complex(p.f_hat, p.N_total, "adjoint nfft, vector f_hat")

    # finalise the one dimensional plan
    nfft_finalize(&p)


def simple_test_nfft_2d():
    cdef int K, N[2], n[2], k, M
    cdef double t

    cdef nfft_plan p

    N[0]=32
    n[0]=64
    N[1]=14
    n[1]=32
    M=N[0]*N[1]
    K=16

    t=nfft_second()
    # init a two dimensional plan
    nfft_init_guru(&p, 2, N, M, n, 7,
        PRE_PHI_HUT| PRE_FULL_PSI| MALLOC_F_HAT| MALLOC_X| MALLOC_F |
        FFTW_INIT| FFT_OUT_OF_PLACE,
        FFTW_ESTIMATE| FFTW_DESTROY_INPUT)

    # init pseudo random nodes
    nfft_vrand_shifted_unit_double(p.x,p.d*p.M_total)

    # precompute psi, the entries of the matrix B
    if p.nfft_flags & PRE_ONE_PSI:
        nfft_precompute_one_psi(&p)

    # init pseudo random Fourier coefficients and show them
    nfft_vrand_unit_complex(p.f_hat,p.N_total)

    t=nfft_second()-t
    nfft_vpr_complex(p.f_hat,K,
        "given Fourier coefficients, vector f_hat (first few entries)")
    printf(" ... initialisation took %e seconds.",t)

    # direct trafo and show the result
    t=nfft_second()
    ndft_trafo(&p)
    t=nfft_second()-t
    nfft_vpr_complex(p.f,K,"ndft, vector f (first few entries)")
    printf(" took %e seconds.",t)

    # approx. trafo and show the result
    t=nfft_second()
    nfft_trafo(&p)
    t=nfft_second()-t
    nfft_vpr_complex(p.f,K,"nfft, vector f (first few entries)")
    printf(" took %e seconds.",t)

    # direct adjoint and show the result
    t=nfft_second()
    ndft_adjoint(&p)
    t=nfft_second()-t
    nfft_vpr_complex(p.f_hat,K,"adjoint ndft, vector f_hat (first few entries)")
    printf(" took %e seconds.",t)

    # approx. adjoint and show the result
    t=nfft_second()
    nfft_adjoint(&p)
    t=nfft_second()-t
    nfft_vpr_complex(p.f_hat,K,"adjoint nfft, vector f_hat (first few entries)")
    printf(" took %e seconds.",t)

    # finalise the two dimensional plan
    nfft_finalize(&p)

def main():
    printf("\n\n1) computing an one dimensional ndft, nfft and an adjoint nfft")
    simple_test_nfft_1d()

    printf("\n\n2) computing a two dimensional ndft, nfft and an adjoint nfft");
    simple_test_nfft_2d()

main()
