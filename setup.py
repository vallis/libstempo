import os
import platform
import subprocess
import warnings
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


# we assume that you have either installed tempo2 via install_tempo2.sh in the default location
# or you have installed in the usual /usr/local
# or you have set the TEMPO2_PREFIX environment variable
# or the tempo2 executable is in your path
def _get_tempo2_install_location():

    # from environment variable
    tempo2_environ = os.getenv("TEMPO2_PREFIX")
    if tempo2_environ is not None:
        return tempo2_environ

    # first check local install
    local = Path(os.getenv("HOME")) / ".local"
    if (local / "include/tempo2.h").exists():
        return str(local)

    # next try global
    glbl = Path("/usr/local")
    if (glbl / "include/tempo2.h").exists():
        return str(glbl)

    # if not, check for tempo2 binary in path
    try:
        out = subprocess.check_output("which tempo2", shell=True)
        out = out.decode().strip()
    except subprocess.CalledProcessError:
        warnings.warn(("tempo2 does not appear to be in your path."))
    else:
        # the executable should be in in bin/ so navigate back and check include/
        root_dir = Path(out).parents[1]
        if (root_dir / "include/tempo2.h").exists():
            return str(root_dir)

    raise RuntimeError(
        """
        Cannot find tempo2 install location. Your options are:

        1. Use the install_tempo2.sh script without any arguments to install to default location.
        2. Install tempo2 globally in /usr/local
        3. Set the TEMPO2_PREFIX environment variable:
            For example, if the tempo2 executable lives in /opt/local/bin:
                TEMPO2_PREFIX=/opt/local pip install libstempo
                or
                export TEMPO2_PREFIX=/opt/local
                pip install libstempo
        """
    )


TEMPO2 = _get_tempo2_install_location()

# need rpath links to shared libraries on Linux
if platform.system() == "Linux":
    linkArgs = ["-Wl,-R{}/lib".format(TEMPO2)]
else:
    linkArgs = []

setup(
    name="libstempo",
    version="2.4.2",  # remember to change it in __init__.py
    description="A Python wrapper for tempo2",
    author="Michele Vallisneri",
    author_email="vallis@vallis.org",
    url="https://github.com/vallis/libstempo",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=["libstempo"],
    package_dir={"libstempo": "libstempo"},
    package_data={"libstempo": ["data/*", "ecc_vs_nharm.txt"]},
    py_modules=[
        "libstempo.like",
        "libstempo.multinest",
        "libstempo.emcee",
        "libstempo.plot",
        "libstempo.toasim",
        "libstempo.spharmORFbasis",
        "libstempo.eccUtils",
    ],
    install_requires=["Cython>=0.22", "numpy>=1.15.0", "scipy>=1.2.0", "matplotlib>=3.3.2", "ephem>=3.7.7.1"],
    extras_require={"astropy": ["astropy>=4.1"]},
    python_requires=">=3.6",
    ext_modules=cythonize(
        Extension(
            "libstempo.libstempo",
            ["libstempo/libstempo.pyx"],
            language="c++",
            include_dirs=[TEMPO2 + "/include", numpy.get_include()],
            libraries=["tempo2", "tempo2pred"],
            library_dirs=[TEMPO2 + "/lib"],
            extra_compile_args=["-Wno-unused-function"],
            extra_link_args=linkArgs,
        )
    ),
)
