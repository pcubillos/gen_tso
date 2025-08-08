# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'ROOT',
    'KNOWN_PROGRAMS',
    'check_latest_version',
    'get_latest_pandeia_release',
    'get_version_advice',
    'get_pandeia_advice',
    'read_spectrum_file',
    'collect_spectra',
    'format_text',
    'pretty_print_target',
]

import os
from packaging.version import parse

from bs4 import BeautifulSoup
import numpy as np
import pyratbay.constants as pc
import pyratbay.tools as pt
import requests
from shiny import ui

ROOT = os.path.realpath(os.path.dirname(__file__)) + '/'
from .catalogs.utils import as_str


# Manually kept:
KNOWN_PROGRAMS = [
    1033, 1118, 1177, 1185, 1201, 1224, 1274, 1279, 1280, 1281, 1312,
    1331, 1353, 1366, 1442, 1541, 1633, 1729, 1743, 1803, 1846, 1935,
    1952, 1981, 2001, 2008, 2021, 2055, 2062, 2084, 2113, 2149, 2158,
    2159, 2304, 2319, 2334, 2347, 2358, 2372, 2420, 2437, 2454, 2488,
    2498, 2507, 2508, 2512, 2571, 2589, 2594, 2667, 2708, 2722, 2734,
    2759, 2765, 2783, 2950, 2961, 3077, 3154, 3171, 3231, 3235, 3263,
    3279, 3315, 3385, 3557, 3615, 3712, 3730, 3731, 3784, 3818, 3838,
    3860, 3942, 3969, 4008, 4082, 4098, 4102, 4105, 4126, 4195, 4227,
    4536, 4711, 4818, 4931, 5022, 5177, 5191, 5268, 5311, 5531, 5634,
    5687, 5799, 5844, 5863, 5866, 5882, 5894, 5924, 5959, 5967, 6045,
    6193, 6284, 6456, 6457, 6491, 6543,
    6932, 6978, 7073, 7188, 7251, 7255, 7407, 7675, 7683, 7686, 7849,
    7875, 7953, 7982, 8004, 8017, 8233, 8309, 8597, 8696, 8739, 8864,
    8877, 9025, 9033, 9095, 9101, 9235,
]


def check_latest_version(package):
    response = requests.get(f'https://pypi.org/pypi/{package}/json')
    latest_version = response.json()['info']['version']
    return latest_version


def get_latest_pandeia_release():
    """
    Fetch latest pandeia.engine version for JWST from their website
    """
    url = 'https://outerspace.stsci.edu/display/PEN/Pandeia+Engine+News'
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; MyBot/0.1; +https://example.com/bot)'
    }
    try:
        response = requests.get(url, headers=headers)
        status_code = response.status_code
    except:
        status_code = 0
    if status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        for a in soup.find_all('a', href=True):
            link = 'pypi.org/project/pandeia.engine/'
            if link in a['href']:
                href = a['href']
                ini = href.index(link) + len(link)
                version = href[ini:]
                if '/' in version:
                    version = version[0:version.index('/')]
                return version.strip()
    # Fail-safe, hard-coded JWST default, need to be kept up to date manually:
    return '2025.7'


def get_version_advice(package, latest_version=None):
    name = package.__name__
    my_version = parse(package.__version__)
    if latest_version is None:
        latest_version = parse(check_latest_version(name))
    else:
        latest_version = parse(latest_version)
    my_major_minor = parse(f'{my_version.major}.{my_version.minor}')
    latest_major_minor = parse(f'{latest_version.major}.{latest_version.minor}')
    if my_version == latest_version:
        color = '#0B980D'
        advice = ''
    elif my_major_minor == latest_major_minor:
        color = '#ffa500'
        advice = (
            f'.<br>You may want to upgrade {name} with:<br>'
            f'<span style="font-weight:bold;">pip install --upgrade {name}</span>'
        )
    else:
        color = 'red'
        advice = (
            f'.<br>You should upgrade {name} with:<br>'
            f'<span style="font-weight:bold;">pip install --upgrade {name}</span>'
        )

    status_advice = ui.HTML(
        f'<br><p><span style="color:{color}">You have {name} '
        f'version {my_version}, the latest version is '
        f'{latest_version}</span>{advice}</p>'
    )
    return status_advice


def get_pandeia_advice(package, latest_version):
    name = package.__name__
    my_version = parse(package.__version__)
    latest_version = parse(latest_version)
    if my_version >= latest_version:
        color = '#0B980D'
        advice = ''
    else:
        color = 'red'
        advice = (
            f'.<br>You should upgrade {name} with:<br>'
            f'<span style="font-weight:bold;">pip install --upgrade {name}</span>'
        )

    status_advice = ui.HTML(
        f'<br><p><span style="color:{color}">You have {name} '
        f'version {my_version}, the latest JWST version is '
        f'{latest_version}</span>{advice}</p>'
    )
    return status_advice


def read_spectrum_file(file, units='none', on_fail=None):
    """
    Parameters
    ----------
    file: String
        Spectrum file to read (transit depth, eclipse depth, or stellar SED)
        This is a plain-text file with two columns (white space separater)
        First column is the wavelength, second is the depth/flux.
        Should be readable by numpy.loadtxt().
    units: String
        Units of the input spectrum.
        For depths, use one of 'none', 'percent', 'ppm'.
        For SEDs, use one of
            'mJy', 
            'f_freq' (for erg s-1 cm-2 Hz-1),
            'f_nu' (for for erg s-1 cm-2 cm),
            'f_lambda' (for erg s-1 cm-2 cm-1)
    on_fail: String
        if 'warning' raise a warning.
        if 'error' raise an error.

    Examples
    --------
    >>> import gen_tso.utils as u

    >>> file = f'{u.ROOT}data/models/WASP80b_transit.dat'
    >>> label, wl, spectra = u.read_spectrum_file(file, on_fail='warning')
    """
    # Validate units
    depth_units = [
        "none",
        "percent",
        "ppm",
    ]
    sed_units = [
        "f_freq",
        "f_nu",
        "f_lambda",
        "mJy",
    ]
    if units not in depth_units and units not in sed_units:
        msg = (
            f"The input units ({repr(units)}) must be one of {depth_units} "
            f"for depths or one of {sed_units} for SEDs"
        )
        raise ValueError(msg)

    # (try to) load the file:
    try:
        wl, spectrum = np.loadtxt(file, unpack=True)
    except ValueError as error:
        error_msg = (
            f'Error, could not load spectrum file: {repr(file)}\n{error}'
        )
        if on_fail == 'warning':
            print(error_msg)
        if on_fail == 'error':
            raise ValueError(error_msg)
        return None, None, None

    # Set the units:
    if units in depth_units:
        u = pt.u(units)
    else:
        if units == 'f_freq':
            u = 10**26
        elif units == 'f_nu':
            u = 10**26 / pc.c
        elif units == 'f_lambda':
            u = 10**26 / pc.c * (wl*pc.um)**2.0
        elif 'mJy' in units:
            u = 1.0

    path, label = os.path.split(file)
    if label.endswith('.dat') or label.endswith('.txt'):
        label = label[0:-4]
    return label, wl, spectrum*u


def collect_spectra(folder, on_fail=None):
    """
    Collect transit, eclipse, and SED spectra files from folder.
    - Transit spectra are identified by having 'transmission' or 'transit'
    in their names.  Or all files contained in a 'transit/' subfolder.
    - Eclipse spectra are identified by having 'emission' or 'eclipse'
    in their names.  Or all files contained in a 'eclipse/' subfolder.
    - SED spectra are identified by having 'sed' or 'star' in their names.

    Parameters
    ----------
    folder: String
        The folder where to search.
    on_fail: String
        if 'warning' raise a warning.
        if 'error' raise an error.

    Returns
    -------
    transit_spectra: Dictionary
        Spectrum name: 1D spectrum pairs for each transit file.
            The 1D spectra is itself a dictionary with keys: wl and depth.
    eclipse_spectra: Dictionary
        Spectrum name: 1D spectrum pairs for each eclipse file.
            The 1D spectra is itself a dictionary with keys: wl and depth.
    sed_spectra: Dictionary
        Spectrum name: 1D spectrum pairs for each SED file.
            The 1D spectra is itself a dictionary with keys: wl and flux.

    Examples
    --------
    >>> import gen_tso.utils as u

    >>> folder = f'{u.ROOT}data/models/'
    >>> spectra = u.collect_spectra(folder, on_fail=None)
    """
    files = sorted(os.listdir(folder))
    transit_files = [
        file for file in files
        if 'transit' in file or 'transmission' in file
        if not os.path.isdir(f'{folder}/{file}')
    ]
    if 'transit' in files and os.path.isdir(f'{folder}/transit'):
        sub_folder = f'{folder}/transit'
        transit_files += [
            f'transit/{file}' for file in sorted(os.listdir(sub_folder))
            if not os.path.isdir(f'{folder}/transit/{file}')
        ]

    eclipse_files = [
        file for file in files
        if 'eclipse' in file or 'emission' in file
        if not os.path.isdir(f'{folder}/{file}')
    ]
    if 'eclipse' in files and os.path.isdir(f'{folder}/eclipse'):
        sub_folder = f'{folder}/eclipse'
        eclipse_files += [
            f'eclipse/{file}' for file in sorted(os.listdir(sub_folder))
            if not os.path.isdir(f'{folder}/eclipse/{file}')
        ]

    sed_files = [
        file for file in files
        if 'sed' in file or 'star' in file
        if not os.path.isdir(f'{folder}/{file}')
    ]
    if 'sed' in files and os.path.isdir(f'{folder}/sed'):
        sub_folder = f'{folder}/sed'
        sed_files += [
            f'sed/{file}' for file in sorted(os.listdir(sub_folder))
            if not os.path.isdir(f'{folder}/sed/{file}')
        ]

    transit_spectra = {}
    for file in transit_files:
        filename = f'{folder}/{file}'
        units = 'none'
        label, wl, depth = read_spectrum_file(filename, units, on_fail)
        if wl is not None:
            transit_spectra[label] = {
                'wl': wl,
                'depth': depth,
                'units': units,
                'filename': filename,
            }

    eclipse_spectra = {}
    for file in eclipse_files:
        filename = f'{folder}/{file}'
        units = 'none'
        label, wl, depth = read_spectrum_file(filename, units, on_fail)
        if wl is not None:
            eclipse_spectra[label] = {
                'wl': wl,
                'depth': depth,
                'units': units,
                'filename': filename,
            }

    sed_spectra = {}
    for file in sed_files:
        filename = f'{folder}/{file}'
        units = 'mJy'
        label, wl, model = read_spectrum_file(filename, units, on_fail)
        if wl is not None:
            sed_spectra[label] = {
                'wl': wl,
                'flux': model,
                'units': units,
                'filename': filename,
            }

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
        If None return plain text.
        If 'html' return HTML formatted text.
        If 'rich' return formatted text to be printed with prompt_toolkit.

    See also
    --------
    gen_tso.pandeia_io.tso_print

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
    elif warning:
        status = 'warning'

    if format is None or status=='normal':
        return text

    if format == 'html':
        text_value = f'<span class="{status}">{text}</span>'
    elif format == 'rich':
        text_value = f'<{status}>{text}</{status}>'
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

    color = '#15B01A' if target.is_jwst_planet else 'black'
    jwst_planet = f'<span style="color:{color}">{target.is_jwst_planet}</span>'
    color = '#15B01A' if target.is_jwst_host else 'black'
    jwst_host = f'<span style="color:{color}">{target.is_jwst_host}</span>'

    planet_info = ui.HTML(
        f'planet = {repr(target.planet)}<br>'
        f'is_jwst_planet = {jwst_planet}<br>'
        f'is_transiting = {target.is_transiting}<br>'
        f"status = '{status} planet'<br><br>"
        f"rplanet = {rplanet} r_earth<br>"
        f"{mplanet_label} = {mplanet} m_earth<br>"
        f"semi_major_axis = {sma} AU<br>"
        f"period = {period} d<br>"
        f"equilibrium_temp = {eq_temp} K<br>"
        f"transit_duration = {t_dur} h<br>"
        f"rplanet/rstar = {rprs}<br>"
        f"a/rstar = {ars}<br>"
    )

    star_info = ui.HTML(
        f'host = {repr(target.host)}<br>'
        f'is_jwst_host = {jwst_host}<br>'
        f'<br><br><br>'
        f"rstar = {rstar} r_sun<br>"
        f"mstar = {mstar} m_sun<br>"
        f"log_g = {logg}<br>"
        f"metallicity = {metal}<br>"
        f"effective_temp = {teff} K<br>"
        f"Ks_mag = {ks_mag}<br>"
        f"RA = {target.ra:.3f} deg<br>"
        f"dec = {target.dec:.3f} deg<br>"
    )

    return planet_info, star_info, aliases



