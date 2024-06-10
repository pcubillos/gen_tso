# Copyright (c) 2021-2024 Patricio Cubillos
# Pyrat Bay is open-source software under the GPL-2.0 license
# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

"""
Some of these routines have been lifted from the pyratbay package
When pyratbay 2.0 gets release, remove these and add dependency.
"""

__all__ = [
    'ROOT',
    'Tophat',
    'constant_resolution_spectrum',
    'bin_spectrum',
    'read_spectrum_file',
    'collect_spectra',
    'format_text',
    'pretty_print_target',
]

import operator
import os

import numpy as np
from prompt_toolkit.formatted_text import FormattedText
from pyratbay.tools import u
from pyratbay.spectrum import PassBand
from shiny import ui

ROOT = os.path.realpath(os.path.dirname(__file__)) + '/'
from .catalogs.utils import as_str


class Tophat(PassBand):
    """
    A Filter passband object with a tophat-shaped passband.
    """
    def __init__(
            self, wl0, half_width,
            name='tophat', ignore_gaps=False,
        ):
        """
        Parameters
        ----------
        wl0: Float
            The passband's central wavelength (um units).
        half_width: Float
            The passband's half-width (um units).
        name: Str
            A user-defined name for the filter when calling str(self),
            e.g., to identify the instrument provenance of this filter.

        Examples
        --------
        >>> import pyratbay.spectrum as ps
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np

        >>> hat = ps.Tophat(4.5, 0.5)

        >>> # Evaluate over a wavelength array (um units):
        >>> wl = np.arange(3.5, 5.5, 0.001)
        >>> out_wl, out_response = hat(wl)

        >>> plt.figure(1)
        >>> plt.clf()
        >>> plt.plot(out_wl, out_response)
        >>> plt.plot(hat.wl, hat.response)  # Same variables
        >>> # Note wl differs from hat.wl, but original array can be used as:
        >>> plt.plot(wl[hat.idx], hat.response)

        >>> # Evaluate over a wavenumber array:
        >>> wn = 1e4 / wl
        >>> hat(wn=wn)
        >>> plt.figure(1)
        >>> plt.clf()
        >>> plt.plot(hat.wn, hat.response, dashes=(5,3))
        >>> plt.plot(wn[hat.idx], out_response)
        """
        self.wl0 = wl0
        self.half_width = half_width
        # Read filter wavenumber and transmission curves:
        self.wn0 = 1.0 / (self.wl0 * u('um'))
        self.ignore_gaps = ignore_gaps

        self.name = name

    def __repr__(self):
        return f'pyratbay.spectrum.Tophat({self.wl0}, {self.half_width})'

    def __str__(self):
        return f'{self.name}_{self.wl0}um'

    def __call__(self, wl=None, wn=None):
        """
        Interpolate filter response function at specified spectral array.
        The response funciton is normalized such that the integral over
        wavenumber equals one.

        Parameters
        ----------
        wl: 1D float array
            Wavelength array at which evaluate the passband response's
            in micron units.
            (only one of wl or wn should be provided on call)
        wn: 1D float array
            Wavenumber array at which evaluate the passband response's
            in cm-1 units.
            (only one of wl or wn should be provided on call)
        ignore_gaps: Bool
            If True, set variables to None when there are no points
            inside the band. A ps.bin_spectrum() call on such band
            will return np.nan.
            Otherwise the code will throw an IndexError when attempting to
            index from an empty array.

        Defines
        -------
        self.response  Normalized interpolated response function
        self.idx       Wavenumber indices
        self.wn        Passband's wavenumber array
        self.wl        Passband's wavelength array

        Returns
        -------
        out_wave: 1D float array
            Same as self.wl or self.wn depending on the input argument.
        out_response: 1D float array
            Same as self.response

        Examples
        --------
        >>> # See examples in help(ps.Tophat.__init__)
        """
        if not operator.xor(wl is None, wn is None):
            raise ValueError(
                'Either provide wavelength or wavenumber array, not both'
            )
        input_is_wl = wn is None
        if input_is_wl:
            wn = 1.0 / (wl*u('um'))

        sign = np.sign(np.ediff1d(wn))
        if not (np.all(sign == 1) or np.all(sign == -1)):
            raise ValueError(
                'Input wavelength/wavenumber array must be strictly '
                'increasing or decreasing'
            )
        sign = sign[0]
        nwave = len(wn)

        wl_low  = self.wl0 - self.half_width
        wl_high = self.wl0 + self.half_width

        wn_low = 1.0 / (wl_high*u('um'))
        wn_high = 1.0 / (wl_low*u('um'))
        idx = (wn >= wn_low) & (wn <= wn_high)
        indices = np.where(idx)[0]

        if self.ignore_gaps and len(indices) == 0:
            self.idx = None
            self.response = None
            self.wn = None
            self.wl = None
            return None, None

        # One spectral point as margin:
        idx_first = indices[0]
        idx_last = indices[-1] + 1

        if idx_first > 0:
            idx_first -= 1
        if idx_last < nwave:
            idx_last += 1

        if sign < 0.0:
            self.idx = np.flip(np.arange(idx_first, idx_last))
        else:
            self.idx = np.arange(idx_first, idx_last)

        self.wn = wn[self.idx]
        self.response = np.array(idx[self.idx], np.double)
        self.response /= np.trapz(self.response, self.wn)

        self.wl = 1.0 / (self.wn * u('um'))
        if input_is_wl:
            out_wave = self.wl
        else:
            out_wave = self.wn

        return out_wave, self.response


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


def bin_spectrum(bin_wl, wl, spectrum, half_widths=None, ignore_gaps=False):
    """
    Bin down a spectrum.

    Parameters
    ----------
    bin_wl: 1D float array
        Central wavelength (um) of the desired binned down spectra.
    wl: 1D float array
        Wavelength samples of the original spectrum.
    spectrum: 1D float array
        Spectral values to be binned down.
    half_widths: 1D float array
        The bin half widths (um).
        If None, assume that the bin edges are at the mid-points
        of the bin_wl array.

    Returns
    -------
    bin_spectrum: 1D float array
        The binned spectrum.

    Notes
    -----
    Probably bad things will happen if bin_wl has a similar
    or coarser resolution than wl.

    Examples
    --------
    >>> import pyratbay.spectrum as ps
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> # Make a noisy high-resolution signal
    >>> wl = ps.constant_resolution_spectrum(1.0, 3.0, resolution=5000)
    >>> spectrum = np.sin(3.14*wl) + np.random.normal(1.5, 0.1, len(wl))
    >>> # Bin it down:
    >>> bin_wl = ps.constant_resolution_spectrum(1.0, 3.0, resolution=125)
    >>> bin_spectrum = ps.bin_spectrum(bin_wl, wl, spectrum)

    >>> # Compare original and binned signals
    >>> plt.figure(0)
    >>> plt.clf()
    >>> plt.plot(wl, spectrum, '.', ms=2, color='gray')
    >>> plt.plot(bin_wl, bin_spectrum, color='red')
    """
    if half_widths is None:
        half_widths = np.ediff1d(bin_wl, 0, 0)
        half_widths[0] = half_widths[1]
        half_widths[-1] = half_widths[-2]
        half_widths /= 2.0
    bands = [
        Tophat(wl0, half_width, ignore_gaps=ignore_gaps)
        for wl0, half_width in zip(bin_wl, half_widths)
    ]
    nbands = len(bands)
    band_flux = np.zeros(nbands)
    for i,band in enumerate(bands):
        band_wl, response = band(wl)
        if band.idx is None:
            band_flux[i] = np.nan
        else:
            band_flux[i] = np.trapz(spectrum[band.idx]*response, band.wn)
    return band_flux


def read_spectrum_file(file, on_fail=None):
    """
    Parameters
    ----------
    file: String
        Spectrum file to read (transit depth, eclipse depth, or stellar SED)
        This is a plain-text file with two columns (white space separater)
        First column is the wavelength, second is the depth/flux.
        Should be readable by numpy.loadtxt().
    on_fail: String
        if 'warning' raise a warning.
        if 'error' raise an error.

    Examples
    --------
    >>> import gen_tso.utils as u

    >>> file = f'{u.ROOT}data/models/WASP80b_transit.dat'
    >>> spectra = u.read_spectrum_file(file, on_fail='warning')
    """
    try:
        data = np.loadtxt(file, unpack=True)
        wl, depth = data
    except ValueError as error:
        wl = None
        depth = None
        str_error = str(on_fail).capitalize()
        error_msg = (
                f'{str_error}, could not load spectrum file: '
                f'{repr(file)}\n'
                f'{error}'
        )
        if on_fail == 'warning':
            print(error_msg)
        if on_fail == 'error':
            raise ValueError(error_msg)

    path, label = os.path.split(file)
    if label.endswith('.dat') or label.endswith('.txt'):
        label = label[0:-4]
    return label, wl, depth


def collect_spectra(folder, on_fail=None):
    """
    Parameters
    ----------
    on_fail: String
        if 'warning' raise a warning.
        if 'error' raise an error.
    Examples
    --------
    >>> import gen_tso.utils as u

    >>> folder = f'{u.ROOT}data/models/'
    >>> spectra = u.collect_spectra(folder, on_fail=None)
    """
    files = os.listdir(folder)
    transit_files = [
        file for file in sorted(files)
        if 'transit' in file or 'transmission' in file
    ]
    eclipse_files = [
        file for file in sorted(files)
        if 'eclipse' in file or 'emission' in file
    ]
    sed_files = [
        file for file in sorted(files)
        if 'sed' in file or 'star' in file
    ]

    transit_spectra = {}
    for file in transit_files:
        label, wl, depth = read_spectrum_file(f'{folder}/{file}', on_fail)
        if wl is not None:
            transit_spectra[label] = {'wl': wl, 'depth': depth}

    eclipse_spectra = {}
    for file in eclipse_files:
        label, wl, depth = read_spectrum_file(f'{folder}/{file}', on_fail)
        if wl is not None:
            eclipse_spectra[label] = {'wl': wl, 'depth': depth}

    sed_spectra = {}
    for file in sed_files:
        label, wl, model = read_spectrum_file(f'{folder}/{file}', on_fail)
        if wl is not None:
            sed_spectra[label] = {'wl': wl, 'depth': model}

    return transit_spectra, eclipse_spectra, sed_spectra


def format_text(text, warning=False, danger=False, format=None):
    """
    Return a colorful text depending on requested format and warning
    or danger flags.

    Parameters
    ----------
    text: String
        A text to print with optional richer format.
    warning: Bool
        If True, format as warning text (orange color).
    danger: Bool
        If True, format as danger text (red color).
        If True, overrides warning.
    format: String
        Leave as None for plain text. Set to 'html' for HTML format.
        Set to 'rich' for prompt_toolkit FormattedText.

    Examples
    --------
    >>> import gen_tso.utils as u
    >>> text = 'WASP-80 b'
    >>> plain = u.format_text(text, danger=True)
    >>> normal = u.format_text(text, warning=False, danger=False, format='html')
    >>> html = u.format_text(text, danger=True, format='html')
    >>> rich = u.format_text(text, danger=True, format='rich')

    >>> warned = u.format_text(text, warning=True, format='html')
    >>> danger1 = u.format_text(text, danger=True, format='html')
    >>> danger2 = u.format_text(text, warning=True, danger=True, format='html')
    """
    status = 'normal'
    if danger:
        status = 'danger'
        color = '#cb2222'
    elif warning:
        status = 'warning'
        color = '#ffa500'

    if format is None or status=='normal':
        return text

    if format == 'html':
        text_value = f'<span class="{status}">{text}</span>'
    elif format == 'rich':
        text_value = FormattedText([(color, text)])
    return text_value


def pretty_print_target(target):
    """
    Print a target's info to HTML text.
    Must look pretty.
    """
    rplanet = as_str(target.rplanet, '.3f', '---')
    mplanet = as_str(target.mplanet, '.3f', '---')
    sma = as_str(target.sma, '.3f', '---')
    rprs = as_str(target.rprs, '.3f', '---')
    ars = as_str(target.ars, '.3f', '---')
    period = as_str(target.period, '.3f', '---')
    t_dur = as_str(target.transit_dur, '.3f', '---')
    eq_temp = as_str(target.eq_temp, '.1f', '---')

    rstar = as_str(target.rstar, '.3f', '---')
    mstar = as_str(target.mstar, '.3f', '---')
    logg = as_str(target.logg_star, '.2f', '---')
    metal = as_str(target.metal_star, '.2f', '---')
    teff = as_str(target.teff, '.1f', '---')
    ks_mag = as_str(target.ks_mag, '.2f', '---')

    status = 'confirmed' if target.is_confirmed else 'candidate'
    mplanet_label = 'M*sin(i)' if target.is_min_mass else 'mplanet'
    if len(target.aliases) > 0:
        aliases = f'aliases = {target.aliases}'
    else:
        aliases = ''

    planet_info = ui.HTML(
        f'planet = {target.planet} <br>'
        f'is_transiting = {target.is_transiting}<br>'
        f'status = {status} planet<br><br>'
        f"rplanet = {rplanet} r_earth<br>"
        f"{mplanet_label} = {mplanet} m_earth<br>"
        f"semi-major axis = {sma} AU<br>"
        f"period = {period} d<br>"
        f"equilibrium temp = {eq_temp} K<br>"
        f"transit_dur (T14) = {t_dur} h<br>"
        f"rplanet/rstar = {rprs}<br>"
        f"a/rstar = {ars}<br>"
    )

    star_info = ui.HTML(
        f'host = {target.host}<br>'
        f'is JWST host = {target.is_jwst}<br>'
        f'<br><br>'
        f"rstar = {rstar} r_sun<br>"
        f"mstar = {mstar} m_sun<br>"
        f"log_g = {logg}<br>"
        f"metallicity = {metal}<br>"
        f"effective temp = {teff} K<br>"
        f"Ks_mag = {ks_mag}<br>"
        f"RA = {target.ra:.3f} deg<br>"
        f"dec = {target.dec:.3f} deg<br>"
    )

    return planet_info, star_info, aliases



