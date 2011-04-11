# Make NFFT cython wrapper

test.so: test.c
	python setup.py build_ext --inplace
test.c: test.pyx
	cython test.pyx


# trivial cythonization of simple_test.c:

simple_test.so: simple_test.c
	python setup.py build_ext --inplace
simple_test.c: simple_test.pyx
	cython simple_test.pyx


# simple_test_np and nfftpy are in progress, not working yet:

simple_test_np.so: simple_test_np.c
	python setup.py build_ext --inplace
simple_test_np.c: simple_test_np.pyx
	cython simple_test_np.pyx

nfftpy.so: nfftpy.c
	python setup.py build_ext --inplace
nfftpy.c: nfftpy.pyx
	cython nfftpy.pyx



# non-build commands:

test:
	python -c "import simple_test"

testtest:
	python -c "import test"


clean:
	rm -f test.so test.c
	rm -f simple_test.so simple_test.c
	rm -f simple_test_np.so simple_test_np.c
	rm -f nfftpy.so nfftpy.c
	rm -rf build/
