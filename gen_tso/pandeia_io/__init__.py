# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

# Suppress synphot warnings that no longer apply
import warnings
warnings.filterwarnings("ignore", message=".*Failed to load Vega spectrum.*")

from .pandeia_interface import *
from .pandeia_calculation import *
from .pandeia_defaults import *

__all__ = (
    pandeia_interface.__all__
    + pandeia_calculation.__all__
    + pandeia_defaults.__all__
)


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

