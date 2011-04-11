# Make NFFT cython wrapper

# tests in progress:

simple_test_np.so: simple_test_np.c
	python setup.py build_ext --inplace
simple_test_np.c: simple_test_np.pyx
	cython simple_test_np.pyx


# trivial cythonization of simple_test.c:

simple_test.so: simple_test.c
	python setup.py build_ext --inplace --simple
simple_test.c: simple_test.pyx
	cython simple_test.pyx


# nfftpy in progress, not working yet:

nfftpy.so: nfftpy.c
	python setup.py build_ext --inplace
nfftpy.c: nfftpy.pyx
	cython nfftpy.pyx


temp.so: temp.c
	python setup.py build_ext --inplace --temp
	echo "\n"
	python -c "import temp"
temp.c: temp.pyx
	cython temp.pyx


# non-build commands:

test:
	python -c "import simple_test_np"

test_simple:
	python -c "import simple_test"

clean:
	rm -f temp.so temp.c
	rm -f simple_test.so simple_test.c
	rm -f simple_test_np.so simple_test_np.c
	rm -f nfftpy.so nfftpy.c
	rm -rf build/
