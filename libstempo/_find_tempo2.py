import logging
import os
import subprocess
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

RUNTIME_DIRS = ("atmosphere", "clock", "earth", "ephemeris", "observatory", "solarWindModel")
HOME = os.getenv("HOME")


def find_tempo2_runtime():
    """
    Attempt to find TEMPO2 runtime if TEMPO2 environment variable is not set
    """

    # first check for local install (i.e. from using install_tempo2.sh)
    local_path = Path(HOME) / ".local/share/tempo2"
    if local_path.exists():
        return str(local_path)

    # if not, check for tempo2 binary in path
    try:
        out = subprocess.check_output("which tempo2", shell=True)
        out = out.decode().strip()
    except subprocess.CalledProcessError:
        warnings.warn("Could not find tempo2 executable in your path")
    else:

        # since this would be in a bin/ directory, navigate back to root and check share/
        share_dir = Path(out).parents[1] / "share"

        if share_dir.exists():
            # loop through all directories in share
            for d in share_dir.iterdir():
                if d.is_dir():
                    # if this directory contains the runtime dirs then set this to be the runtime dir
                    dirs = [dd.stem for dd in d.iterdir() if dd.is_dir()]
                    if all(rd in dirs for rd in RUNTIME_DIRS):
                        return str(d)
    raise RuntimeError("Can't find T2runtime from inspection. Set TEMPO2 environment variable")
