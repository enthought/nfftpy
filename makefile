# Make NFFT cython wrapper

# tests in progress:

simple_test_class.so: simple_test_class.pyx nfftpy.so
	cython simple_test_class.pyx
	python setup.py build_ext --inplace --test simple_test_class

temp.so: temp.pyx
	cython temp.pyx
	python setup.py build_ext --inplace --test temp
	echo "\n"
	python -c "import temp"


# trivial cythonization of simple_test.c, and with numpy arrays:

simple_test.so: simple_test.pyx
	cython simple_test.pyx
	python setup.py build_ext --inplace --test simple_test

simple_test_np.so: simple_test_np.pyx
	cython simple_test_np.pyx
	python setup.py build_ext --inplace --test simple_test_np


# nfftpy in progress, not working yet:

nfftpy.so: nfftpy.pyx
	cython nfftpy.pyx
	python setup.py build_ext --inplace


# non-build commands:

test:
	python -c "import simple_test_class"

test_simple:
	python -c "import simple_test_np"

test_class:
	python -c "import simple_test_class"

clean:
	rm -f *.so *.c
	rm -rf build/
