import os
import platform
import subprocess
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup


# we assume that you have either installed tempo2 via install_tempo2.sh
# or you have installed in the usual /usr/local
# or the tempo2 executable is in your path
def _get_tempo2_install_location():
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
        raise subprocess.CalledProcessError(
            ("tempo2 does not appear to be in your path. Please make sure the executable is in your path")
        )

    # the executable should be in in bin/ so navigate back and check include/
    root_dir = Path(out).parents[1]
    if (root_dir / "include/tempo2.h").exists():
        return str(root_dir)

    raise RuntimeError(
        "Cannot find tempo2 install location. Use install_tempo2.sh script to install or install globally."
    )


TEMPO2 = _get_tempo2_install_location()

# need rpath links to shared libraries on Linux
if platform.system() == "Linux":
    linkArgs = ["-Wl,-R{}/lib".format(TEMPO2)]
else:
    linkArgs = []

setup(
    name="libstempo",
    version="2.3.5",  # remember to change it in __init__.py
    description="A Python wrapper for tempo2",
    author="Michele Vallisneri",
    author_email="vallis@vallis.org",
    url="https://github.com/vallis/libstempo",
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
