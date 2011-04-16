# Make NFFT cython wrapper

all: simple_test_class.so simple_test.so

# cythonization of simple_test.c, but using wrapper class:
simple_test_class.so: simple_test_class.pyx nfftpy.so nfftpy.pxd cnfft3util.pxd
	python setup.py build_ext --inplace --test simple_test_class

# trivial cythonization of simple_test.c:
simple_test.so: simple_test.pyx cnfft3.pxd cnfft3util.pxd
	python setup.py build_ext --inplace --test simple_test

# core wrapper class:
nfftpy.so: nfftpy.pyx cnfft3.pxd
	python setup.py build_ext --inplace


# non-build commands:

test: nfftpy.so
	nosetests -s --verbose

test_simple: simple_test.so
	python -c "import simple_test"

test_simple_class: simple_test_class.so nfftpy.so
	python -c "import simple_test_class"

clean:
	rm -f *.so *.c
	rm -rf build/
