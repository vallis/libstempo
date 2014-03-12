# libstempo — a Python wrapper for tempo2 #

## Installation notes ##

* `libstempo` requires Python 2.7+ (but is currently untested with Python 3), [numpy](http://www.numpy.org/) and [Cython](http://www.cython.org/) (v > 0.19), and of course [tempo2](http://www.atnf.csiro.au/research/pulsar/tempo2/) (v 2013.9.1 or greater).
* If you don't need to look at or modify the sources, you should be able to do simply `pip install libstempo`, and everything will happen for you.
* Otherwise, to install `libstempo` do `python setup.py install --prefix=...` (wherever you normally install). The installer will look in likely places for the `tempo2` headers and libraries, but you can also give it the installation directory using `--with-tempo2=...` (the installation directory would be `XXX` if `tempo2.h` is installed to `XXX/include`).
* If the standard installation does not work, there's a `Makefile` (written for OS X) that you could try.

## Documentation ##

For the moment, _in lieu_ of documentation, have a look at this [tutorial](http://nbviewer.ipython.org/urls/raw.github.com/vallis/mc3pta/master/libstempo/libstempo-demo.ipynb) (an [iPython notebook](http://ipython.org/notebook.html), visualized thanks to the [nbviewer](http://nbviewer.ipython.org/) service).
