#!/usr/bin/env python

import sys, os

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

tempo2 = None

argv_replace = []
for arg in sys.argv:
    if arg.startswith('--with-tempo2='):
        tempo2 = arg.split('=', 1)[1]
    else:
        argv_replace.append(arg)
sys.argv = argv_replace

if tempo2 is None:
    # hmm, you're making things hard, huh? let's try autodetecting in a few likely places

    try:
        import subprocess
        stdout = subprocess.check_output('which tempo2',shell=True)
        t2exec = [stdout[:-12]]     # remove /bin/tempo2
    except:
        t2exec = []

    virtenv = [os.environ['VIRTUAL_ENV']] if 'VIRTUAL_ENV' in os.environ else []
    ldpath  = map(lambda s: s[:-4],os.environ['LD_LIBRARY_PATH'].split(':')) if 'LD_LIBRARY_PATH' in os.environ else []

    paths = t2exec + virtenv + ldpath + [os.environ['HOME'],'/usr/local','/usr']
    found = [path for path in paths if os.path.isfile(path + '/include/tempo2.h')]
    found = list(set(found))    # remove duplicates

    if found:
        tempo2 = found[0]
        print "Found tempo2 install in {0}, will use {1}.".format(found,"it" if len(found) == 1 else tempo2)
    else:
        print """
Sorry, but I need you to point me to the tempo2 install root
(e.g., /usr/local if tempo2.h is in /usr/local/include), using --with-tempo2=...
"""
        sys.exit(1)

print numpy.get_include()

setup(name = 'libstempo',
      version = '1.2.3',
      description = 'A Python wrapper for tempo2',

      author = 'Michele Vallisneri',
      author_email = 'vallis@vallis.org',
      url = 'https://github.com/vallis/mc3pta',

      package_dir = {'libstempo': '.'},

      py_modules = ['libstempo.like','libstempo.multinest','libstempo.emcee','libstempo.plot','libstempo.toasim'],

      ext_modules = cythonize(Extension('libstempo.libstempo',['libstempo.pyx'],language="c++",
                                        include_dirs = [tempo2 + '/include',numpy.get_include()],
                                        libraries = ['tempo2'],
                                        library_dirs = [tempo2 + '/lib']))
      )
