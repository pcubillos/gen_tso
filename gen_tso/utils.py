# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'ROOT',
    'constant_resolution_spectrum',
]

import os
import numpy as np

ROOT = os.path.realpath(os.path.dirname(__file__)) + '/'


def constant_resolution_spectrum(wave_min, wave_max, resolution):
    """
    Compute a constant resolving-power sampling array.

    Parameters
    ----------
    wave_min: Float
        Lower spectral boundary.  This could be either a wavelength
        or a wavenumber. This is agnositc of units.
    wave_max: Float
        Upper spectral boundary.  This could be either a wavelength
        or a wavenumber. This is agnositc of units.
    resolution: Float
        The sampling resolving power: R = wave / delta_wave.

    Returns
    -------
    wave: 1D float array
        A spectrum array with the given resolving power.

    Examples
    --------
    >>> import numpy as np
    >>> import pyratbay.spectrum as ps

    >>> # A low-resolution wavelength sampling:
    >>> wl_min = 0.5
    >>> wl_max = 4.0
    >>> resolution = 5.5
    >>> wl = ps.constant_resolution_spectrum(wl_min, wl_max, resolution)
    >>> print(wl)
    [0.5        0.6        0.72       0.864      1.0368     1.24416
     1.492992   1.7915904  2.14990848 2.57989018 3.09586821 3.71504185]
    >>> # The actual resolution matches the input:
    >>> wl_mean = 0.5*(wl[1:]+wl[:-1])
    >>> print(wl_mean/np.ediff1d(wl))
    [5.5 5.5 5.5 5.5 5.5 5.5 5.5 5.5 5.5 5.5 5.5]
    """
    f = 0.5 / resolution
    g = (1.0+f) / (1.0-f)

    nwave = int(np.ceil(-np.log(wave_min/wave_max) / np.log(g)))
    wave = wave_min * g**np.arange(nwave)
    return wave

