import os
import platform
import subprocess
import warnings
from pathlib import Path

import numpy
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


ext_modules = [
    Extension(
        "libstempo.libstempo",
        ["libstempo/libstempo.pyx"],
        language="c++",
        include_dirs=[TEMPO2 + "/include", numpy.get_include()],
        libraries=["tempo2", "tempo2pred"],
        library_dirs=[TEMPO2 + "/lib"],
        extra_compile_args=["-Wno-unused-function"],
        extra_link_args=linkArgs,
    ),
]

# add language level = 3
for e in ext_modules:
    e.cython_directives = {"language_level": "3"}

setup(
    ext_modules=ext_modules,
)
