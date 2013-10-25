PYTHON_INCLUDE = $(shell python -c "import distutils.sysconfig; print distutils.sysconfig.get_python_inc()")
NUMPY_INCLUDE = $(shell python -c "import numpy; print numpy.get_include()")

# modify to the tempo2 installation directory
TEMPO2=/usr/local

ALL: libstempo.so

# Cython and gcc are required to compile the extension
# try "pip install cython" to get the former
libstempo.cpp: libstempo.pyx
	cython --cplus libstempo.pyx

# the options are appropriate for OS X; for Linux replace
# "-bundle -undefined dynamic_lookup" -> "-shared -fPIC"
# also we're assuming that libtempo2 is installed in /usr/local/lib
libstempo.so: libstempo.cpp
	g++ -O2 -bundle -undefined dynamic_lookup \
	-I$(TEMPO2)/include -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) \
	libstempo.cpp -L$(TEMPO2)/lib -ltempo2 -o libstempo.so

# the compiled Python C extension is left in the local directory
# if needed it can be moved to a site-packages folder
