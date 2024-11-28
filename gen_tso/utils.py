# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'ROOT',
    'check_latest_version',
    'get_latest_pandeia_versions',
    'get_version_advice',
    'read_spectrum_file',
    'collect_spectra',
    'format_text',
    'pretty_print_target',
]

import os
from packaging.version import parse

import numpy as np
import requests
from shiny import ui

ROOT = os.path.realpath(os.path.dirname(__file__)) + '/'
from .catalogs.utils import as_str


def check_latest_version(package):
    response = requests.get(f'https://pypi.org/pypi/{package}/json')
    latest_version = response.json()['info']['version']
    return latest_version


def get_latest_pandeia_versions(package_name='pandeia.engine'):
    """
    Get latest pandeia.engine version for JWST and Roman branches.
    To be checked how new JWST pandeia.engine versions are named.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        versions = sorted([
            parse(version) for version in data['releases'].keys()
        ])
        # split JWST and Roman versions (>2024.X)
        jwst_versions = [
            version for version in versions
            if version.major<2024
        ]
        roman_versions = [
            version for version in versions
            if version.major>=2024
        ]
        return str(jwst_versions[-1]), str(roman_versions[-1])
    except requests.exceptions.RequestException:
        # Hardcoded value, need to be manually updated
        return '4.0', '2024.9.1'



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
        error_msg = (
            f'Error, could not load spectrum file: {repr(file)}\n{error}'
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
    files = os.listdir(folder)
    transit_files = [
        file for file in sorted(files)
        if 'transit' in file or 'transmission' in file
        if not os.path.isdir(file)
    ]
    if 'transit' in files and os.path.isdir('transit'):
        sub_folder = f'{folder}/transit'
        transit_files += [
            f'transit/{file}' for file in sorted(os.listdir(sub_folder))
            if not os.path.isdir(f'transit/{file}')
        ]

    eclipse_files = [
        file for file in sorted(files)
        if 'eclipse' in file or 'emission' in file
        if not os.path.isdir(file)
    ]
    if 'eclipse' in files and os.path.isdir('eclipse'):
        sub_folder = f'{folder}/eclipse'
        transit_files += [
            f'eclipse/{file}' for file in sorted(os.listdir(sub_folder))
            if not os.path.isdir(f'eclipse/{file}')
        ]

    sed_files = [
        file for file in sorted(files)
        if 'sed' in file or 'star' in file
        if not os.path.isdir(file)
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
            sed_spectra[label] = {'wl': wl, 'flux': model}

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



