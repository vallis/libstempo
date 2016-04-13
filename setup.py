#!/usr/bin/env python

# matteo: use subprocess.getoutput if available
#         use os.path.join instead of +

from __future__ import print_function

import sys, os

from setuptools import setup
from setuptools import Extension
#from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

print("""WARNING: The libstempo API has changed substantially (for the better) from
         versions 1.X to 2.X. If you need the older 1.X API, you can get an older libstempo
         from https://pypi.python.org/simple/libstempo, or checkout the libstempo1
         branch on GitHub - https://github.com/vallis/libstempo/tree/libstempo1""")

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
        stdout = subprocess.check_output('which tempo2',shell=True).decode()
        t2exec = [stdout[:-12]]     # remove /bin/tempo2
                                    # may fail if there are strange bytes
    except:
        t2exec = []

    virtenv = [os.environ['VIRTUAL_ENV']] if 'VIRTUAL_ENV' in os.environ else []
    ldpath  = map(lambda s: s[:-4],os.environ['LD_LIBRARY_PATH'].split(':')) if 'LD_LIBRARY_PATH' in os.environ else []

    paths = t2exec + virtenv + ldpath + [os.environ['HOME'],'/usr/local','/usr']
    found = [path for path in paths if os.path.isfile(path + '/include/tempo2.h')]
    found = list(set(found))    # remove duplicates

    if found:
        tempo2 = found[0]
        print("Found tempo2 install in {0}, will use {1}.".format(found,"it" if len(found) == 1 else tempo2))
    else:
        # tempo2 won't be there, but at least it should exist as a directory
        tempo2 = '/usr'
        print("""
I have not been able to autodetect the location of the tempo2 headers and
libraries. Nevertheless, I will proceed with the installation. If you get
errors, please run setup.py again, but use the option --with-tempo2=...
to point me to the tempo2 install root (e.g., /usr/local if tempo2.h is
in /usr/local/include).
""")

setup(name = 'libstempo',
      version = '2.2.5', # remember to change it in __init__.py
      description = 'A Python wrapper for tempo2',

      author = 'Michele Vallisneri',
      author_email = 'vallis@vallis.org',
      url = 'https://github.com/vallis/libstempo',

      packages = ['libstempo'],
      package_dir = {'libstempo': 'libstempo'},
      package_data = {'libstempo': ['data/*', 'ecc_vs_nharm.txt']},

      py_modules = ['libstempo.like','libstempo.multinest','libstempo.emcee',
                    'libstempo.plot','libstempo.toasim',
                    'libstempo.spharmORFbasis', 'libstempo.eccUtils'],

      ext_modules = cythonize(Extension('libstempo.libstempo',['libstempo/libstempo.pyx'],
                                        language="c++",
                                        include_dirs = [tempo2 + '/include',numpy.get_include()],
                                        libraries = ['tempo2'],
                                        library_dirs = [tempo2 + '/lib']))
      )
