# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'catalogs',
    'pandeia_io',
    'plotly_io',
    'custom_shiny',
]

from . import catalogs
from . import pandeia_io
from . import plotly_io
from . import custom_shiny
from .version import __version__


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)

