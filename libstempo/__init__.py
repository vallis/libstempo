from __future__ import absolute_import
__version__ = "2.3.3"

import os
if 'TEMPO2' not in os.environ:
    os.environ['TEMPO2'] = '/usr/local/share/tempo2'

from libstempo.libstempo import *
