__version__ = "2.4.2"

import os
from ._find_tempo2 import find_tempo2_runtime


# check to see if TEMPO2 environment variable is set
TEMPO2_RUNTIME = os.getenv("TEMPO2")


# if not try to find it and raise error otherwise
if not TEMPO2_RUNTIME:
    os.environ["TEMPO2"] = find_tempo2_runtime()


from libstempo.libstempo import *  # noqa F401,F402,F403
