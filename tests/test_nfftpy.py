"""
Unit tests for NFFTPY (cython wrapper for NFFT libraries)
"""
import os
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal, \
     assert_raises

from nfftpy import NfftPlanWrapper, \
    PRE_PHI_HUT, FG_PSI, PRE_LIN_PSI, PRE_FG_PSI, PRE_PSI, PRE_FULL_PSI, \
    MALLOC_X, MALLOC_F_HAT, MALLOC_F, FFT_OUT_OF_PLACE, FFTW_INIT, PRE_ONE_PSI,\
    FFTW_ESTIMATE, FFTW_DESTROY_INPUT


def check_a_plan(pw, x_data, f_hat_data, f_data, adjoint_f_hat_data):
    """
    After a plan is initialized, feed it data, compute transforms,
    and check the results.
    """
    # init pseudo random nodes and check that their values took:
    pw.x = x_data
    _x = pw.x
    assert_array_almost_equal(_x, x_data)

    # precompute psi, the entries of the matrix B
    if pw.nfft_flags & PRE_ONE_PSI:
        pw.nfft_precompute_one_psi()

    # init pseudo random Fourier coefficients and check their values took:
    pw.f_hat = f_hat_data
    _f_hat = pw.f_hat
    assert_array_almost_equal(_f_hat, f_hat_data)

    # direct trafo and test the result
    pw.ndft_trafo()
    _f = pw.f
    assert_array_almost_equal(_f, f_data)

    # approx. trafo and check the result
    # first clear the result array to be sure that it is actually touched.
    pw.f = np.zeros_like(f_data)
    pw.nfft_trafo()
    _f2 = pw.f
    assert_array_almost_equal(_f2, f_data)

    # direct adjoint and check the result
    pw.ndft_adjoint()
    _f_hat2 = pw.f_hat
    assert_array_almost_equal(_f_hat2, adjoint_f_hat_data)

    # approx. adjoint and check the result.
    # first clear the result array to be sure that it is actually touched.
    pw.f_hat = np.zeros_like(f_hat_data)
    pw.nfft_adjoint()
    _f_hat3 = pw.f_hat
    assert_array_almost_equal(_f_hat3, adjoint_f_hat_data)

    # finalise (destroy) the 1D plan
    pw.nfft_finalize()

    # check that instance is no longer usable:
    assert_raises( RuntimeError, pw.nfft_finalize)
    assert_raises( RuntimeError, pw.nfft_trafo)
    assert_raises( RuntimeError, lambda : pw.M_total)


def simple_test_nfft_1d():
    """
    Reproduce NFFT's simple nfft example file, 1d,
    quoting their pseudo-random input data arrays instead of re-creating them.
    """
    # Random data generated as input to simple_test_1d on one system,
    # and the resulting output data. To 2 decimal points, by eye, these
    # match the output from the original simple_test.c
    x_data = np.array([
        -0.49999999999996092,
        -0.49901460532534969,
        -0.45836899840538692,
        -0.32335735745708405,
        -0.13539775160939271,
        -0.40866938788770568,
        -0.40770235230132457,
        -0.012782776053171574,
        0.02675027976210842,
        -0.045566576261755642,
        -0.26682156643606092,
        0.33129178798098735,
        0.43173148197993427,
        0.06805961278714534,
        0.056094332471250397,
        -0.44916808574856759,
        0.2670511597301406,
        -0.48108519652415538,
        -0.24764023804159407])

    f_hat_data = np.array([
        0.29819717337120721+0.53155686462417151j,
        0.92026094185792218+0.81042945352693252j,
        0.1884202504409771+0.57061401932911338j,
        0.07677456488383072+0.98489101762450915j,
        0.11835170935876249+0.78448364392793479j,
        0.10091627339248888+0.019841661979963732j,
        0.37837747421462353+0.68092303834546897j,
        0.75270666116677987+0.62440671501078171j,
        0.12646243436476823+0.7709282624741931j,
        0.18653582159103976+0.5095003917367471j,
        0.31593900281851717+0.36724663196646645j,
        0.87631236498480902+0.52525239838405824j,
        0.42783455812693205+0.66797858971080615j,
        0.1711418296113898+0.86190539909679487j])

    f_data = np.array([
        1.2310658547934172-0.037504013018225563j,
        1.2544603055672601-0.034278664420716101j,
        1.1915216009126144-0.52932168699426541j,
        2.3443193451867002+1.2306410372335743j,
        -0.28361828836507652+0.062669399072910859j,
        -0.36095014283840959-1.8161764363714563j,
        -0.36923961605690142-1.8268972137847443j,
        5.0329759504348459+8.097653659649124j,
        3.0693429106102434+6.8188284506434647j,
        2.7351467422739586+3.6201742290441383j,
        -0.20670179094459115+0.81931413987333968j,
        -0.32232571949426031+0.8599787094803425j,
        -0.3661754994239329-0.89312410514417973j,
        -0.56696016719312103-0.87588228574527838j,
        0.23550238121395164+1.3440560639206356j,
        0.89483404875820272-0.78525645007299139j,
        1.0691302488915426-0.66834814784978436j,
        1.4972605351262145-0.10206962113047104j,
        -0.84631950783724452-0.081291795974373415j])

    adjoint_f_hat_data = np.array([
        -0.21287064007341139+14.229876017076991j,
        13.005313487395096+15.755858156168097j,
        -0.13366119527574941+9.6442632635503891j,
        8.1243605885057022+18.257978069843062j,
        10.82165432736001+24.664248762114305j,
        10.211162341943227+11.600551680878084j,
        3.9503978723453366+24.482475763850154j,
        17.233269191586206+15.203165268425106j,
        4.3831999654204896+21.686724610540175j,
        17.628378530261575+15.069938670755379j,
        2.3467762499829488+14.012183930953785j,
        18.607691532136574+13.836220488003514j,
        1.058377658941543+5.3345535887976006j,
        11.536916090608763+17.501604332336246j])

    N=14
    M=19

    # init a one dimensional plan
    pw = NfftPlanWrapper.nfft_init_1d(N, M)
    assert_equal(pw.M_total, M)
    assert_equal(pw.N_total, N)
    assert_equal(pw.d, 1)

    check_a_plan(pw, x_data, f_hat_data, f_data, adjoint_f_hat_data)


def simple_test_nfft_2d():
    """
    Reproduce NFFT's simple nfft example file, 2d,
    reading their pseudo-random input data arrays from a file
    instead of re-creating them.
    """
    N = np.array([32, 14])
    n = np.array([64, 32])
    M=N.prod()

    # init a two dimensional plan
    pw = NfftPlanWrapper.nfft_init_guru(2, N, M, n, 7,
            PRE_PHI_HUT| PRE_FULL_PSI| MALLOC_F_HAT| MALLOC_X| MALLOC_F |
            FFTW_INIT| FFT_OUT_OF_PLACE,
            FFTW_ESTIMATE| FFTW_DESTROY_INPUT)
    num_x = pw.d * pw.M_total

    assert_equal(pw.M_total, M)
    assert_equal(pw.N_total, M)  # ???? True in this case, but why ????
    assert_equal(pw.d, 2)

    # The data file was created in simple_test_class.pyx.
    # It consists of 4 concatenated arrays, each preceded by
    # its number of elements. The first array is real, the others complex.
    data_filename = os.path.join(os.path.dirname(__file__),
                                 'simple_test_nfft_2d.txt')
    data = np.loadtxt(data_filename, dtype='complex128')
    data_divided = []
    i = 0
    corruption_msg = ('test data in %s is apparently corrupted at row %%i' %
                      data_filename)
    for expected_len in (num_x, pw.N_total, pw.M_total, pw.N_total):
        n_elem = int(round(data[i].real))
        if n_elem != expected_len:
            raise IOError(corruption_msg % i)
        i += 1
        next_elem = i + n_elem
        data_divided.append(data[i : next_elem])
        i += n_elem
    if next_elem != len(data):
        raise IOError(corruption_msg % i)
    x_data, f_hat_data, f_data, adjoint_f_hat_data = data_divided
    x_data = x_data.real.copy()

    check_a_plan(pw, x_data, f_hat_data, f_data, adjoint_f_hat_data)

