#!/usr/bin/env python

# matteo: use subprocess.getoutput if available
#         use os.path.join instead of +

from __future__ import print_function

import sys, os, platform

from setuptools import setup
from setuptools import Extension
import distutils.sysconfig

from Cython.Build import cythonize

import numpy

print("""WARNING: The libstempo API has changed substantially (for the better) from
         versions 1.X to 2.X. If you need the older 1.X API, you can get an older libstempo
         from https://pypi.python.org/simple/libstempo, or checkout the libstempo1
         branch on GitHub - https://github.com/vallis/libstempo/tree/libstempo1""")

tempo2, force_tempo2 = None, False

argv_replace = []
for arg in sys.argv:
    if arg.startswith('--with-tempo2='):
        tempo2 = arg.split('=', 1)[1]
    elif arg.startswith('--force-tempo2-install'):
        force_tempo2 = True
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

    if found and not force_tempo2:
        tempo2 = found[0]
        print("Found tempo2 install in {0}, will use {1}.".format(found,"it" if len(found) == 1 else tempo2))

        if 'TEMPO2' in os.environ:
            runtime = os.environ['TEMPO2']
        else:
            runtime = os.path.join(tempo2,'share','tempo2')
            print("But where is the tempo2 runtime? I'm guessing {}; if I am not right, you should define the environment variable TEMPO2.".format(runtime))
    else:
        # try installing tempo2!
        tempo2 = os.path.dirname(os.path.dirname(os.path.dirname(distutils.sysconfig.get_python_lib())))
        runtime = os.path.join(tempo2,'share','tempo2')

        print("I have not been able to (or I was instructed not to) autodetect the location of the tempo2 headers and libraries.")
        print("I will attempt to download and install tempo2 in {}; runtime files will be in {}.".format(tempo2,runtime))
        print("Please note that if the environment variable TEMPO2 is defined, it will override {}.".format(runtime))

        try:
            subprocess.check_call(["./install_tempo2.sh",tempo2])
        except subprocess.CalledProcessError:
            print("I'm sorry, the tempo2 installation failed. I tried my best!")
            sys.exit(2)

runtime = os.path.join(tempo2,'share','tempo2')
initsrc = open('libstempo/__init__.py.in','r').read().replace("TEMPO2DIR",runtime)
open('libstempo/__init__.py','w').write(initsrc)

# need rpath links to shared libraries on Linux
if platform.system() == 'Linux':
    linkArgs = ['-Wl,-R{}/lib'.format(tempo2)]
else:
    linkArgs = []

setup(name = 'libstempo',
      version = '2.3.1', # remember to change it in __init__.py.in
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
                                        language = "c++",
                                        include_dirs = [tempo2 + '/include',numpy.get_include()],
                                        libraries = ['tempo2','tempo2pred'],
                                        library_dirs = [tempo2 + '/lib'],
                                        extra_compile_args = ["-Wno-unused-function"],
                                        extra_link_args = linkArgs))
      )
