# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

"""
A collection of low-level routines to handle catalogs.
"""

__all__ = [
    'esasky_js_circle',
    'esasky_js_catalog',
    'normalize_name',
    'is_letter',
    'is_candidate',
    'get_letter',
    'get_host',
    'select_alias',
    'invert_aliases',
    'to_float',
    'as_str',
]

import re

import numpy as np


def esasky_js_circle(ra, dec, radius, color='#15B01A'):
    """
    Construct a JS command to draw a circle footprint for ESASky

    Parameters
    ----------
    ra: Float
        Right ascention of the center of the circle footprint (deg).
    dec: Float
        Declination of the center of the circle footprint (deg).
    radius: Float
        Radius of the circle footprint (deg).

    Returns
    -------
    footprint: Dictionary
        A dictionary with the command to draw a circle footprint
        when converted to JSON format, e.g.:
        command = json.dumps(footprint)

    For details on the ESASky JS API see:
    https://www.cosmos.esa.int/web/esdc/esasky-javascript-api

    Examples
    --------
    >>> import gen_tso.catalogs.utils as u
    >>> ra = 315.0259661
    >>> dec = -5.094857
    >>> radius = 80.0
    >>> circle = u.esasky_js_circle(ra, dec, radius)
    """
    footprint = {
        'event': 'overlayFootprints',
        'content': {
            'overlaySet': {
                'type': 'FootprintListOverlay',
                'overlayName': 'visit splitting distance',
                'cooframe': 'J2000',
                'color': color,
                'lineWidth': 5,
                'skyObjectList': [
                    {'name': 'visit splitting distance',
                     'id': 1,
                     'stcs': f'CIRCLE ICRS {ra:.8f} {dec:.8f} {radius/3600:.4f}',
                     'ra_deg': f'{ra:.8f}',
                     'dec_deg': f'{dec:.8f}',
                    }
                ]
            }
        }
    }
    return footprint


def json_target_property(name, value, format):
    """
    Create a json dictionary of a target's property (to be used on
    an overlayCatalog for ESASky).
    """
    prop = {
        'name': name,
        'value': f'{value:{format}}',
        'type': 'STRING'
    }
    return prop


def json_target(index, name, ra, dec, g_mag, teff, logg, separation):
    """
    Create a json dictionary of a target (to be used on an overlayCatalog
    for ESASky).
    """
    data = [
        json_target_property('G mag', g_mag, '.2f'),
        json_target_property('T eff', teff, '.1f'),
        json_target_property('log(g)', logg, '.2f'),
        json_target_property('Separation', separation, '.3f'),
    ]

    target = {
        'name': name,
        'id': index+1,
        'ra': f'{ra:.8f}',
        'dec': f'{dec:.8f}',
        'data': data,
    }
    return target


def esasky_js_catalog(query):
    """
    Construct a JS command to draw an overlayCatalog footprint for ESASky

    Parameters
    ----------
    query: List of arrays
        A list of arrays containing the names, g_mag, teff, logg,
        ra, dec, and separation of a set of targets (see Examples).

    Returns
    -------
    command: Dictionary
        A dictionary with the command to draw an overlayCatalog
        when converted to JSON format, e.g.:
        js_command = json.dumps(command)

    For details on the ESASky JS API see:
    https://www.cosmos.esa.int/web/esdc/esasky-javascript-api

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> import gen_tso.catalogs.utils as u

    >>> # Stellar sources around WASP-69:
    >>> ra_source = 315.0259661
    >>> dec_source = -5.094857
    >>> query = cat.fetch_gaia_targets(ra_source, dec_source)
    >>> circle = u.esasky_js_catalog(query)
    """
    names, g_mag, teff, logg, ra, dec, separation = query
    ntargets = len(names)
    targets = []
    for i in range(ntargets):
        target = json_target(
            i, names[i], ra[i], dec[i],
            g_mag[i], teff[i], logg[i], separation[i],
        )
        targets.append(target)

    command = {
        "event": 'overlayCatalogue',
        'content': {
            'overlaySet': {
                'type': 'SourceListOverlay',
                'overlayName': 'Nearby Gaia sources',
                'cooframe': 'J2000',
                'color': '#ee2345',
                'lineWidth': 10,
                'skyObjectList': targets,
            }
        }
    }
    return command


def normalize_name(target):
    """
    Normalize target names into a 'more standard' format.
    Mainly to resolve trexolists target names.
    """
    name = re.sub(r'\s+', ' ', target)
    # It's a case issue:
    name = name.replace('KEPLER', 'Kepler')
    name = name.replace('TRES', 'TrES')
    name = name.replace('WOLF-', 'Wolf ')
    name = name.replace('HATP', 'HAT-P-')
    name = name.replace('AU-MIC', 'AU Mic')

    # Custom correction before going over prefixes
    if name.startswith('NAME-'):
        name = name[5:]
    # Prefixes
    name = name.replace('GL', 'GJ')
    prefixes = [
        'L', 'G', 'HD', 'GJ', 'LTT', 'LHS', 'HIP', 'WD',
        'LP', '2MASS', 'PSR', 'IRAS', 'TYC', 'TIC', 'PSO',
    ]
    for prefix in prefixes:
        prefix_len = len(prefix)
        if name.startswith(prefix) and not name[prefix_len].isalpha():
            name = name.replace(f'{prefix}-', f'{prefix} ')
            name = name.replace(f'{prefix}_', f'{prefix} ')
            if name[prefix_len] != ' ':
                name = f'{prefix} ' + name[prefix_len:]

    prefixes = ['CD-', 'BD-', 'BD+']
    for prefix in prefixes:
        prefix_len = len(prefix)
        dash_loc = name.find('-', prefix_len)
        if name.startswith(prefix) and dash_loc > 0:
            name = name[0:dash_loc] + ' ' + name[dash_loc+1:]
    # Main star
    if name.endswith('A') and not name[-2].isspace():
        name = name[:-1] + ' A'
    # Planet letter is in the name
    if name.lower().endswith('b') and not name[-2].isalpha():
        name = name[:-1]
    if name.lower().endswith('d') and not name[-2].isalpha():
        name = name[:-1]

    # Custom corrections
    name = name.replace('-offset', '')
    name = name.replace('-updated', '')
    name = name.replace('-copy', '')
    name = name.replace('-revised', '')
    if name.endswith('-'):
        name = name[:-1]

    if name.upper() in ['55CNC', 'RHO01-CNC', '-RHO01-CNC']:
        name = '55 Cnc'
    if name == 'WD 1856':
        name = 'WD 1856+534'
    if 'V1298' in name:
        name = 'V1298 Tau'
    return name


def is_letter(name):
    """
    Check if name ends with a blank + lower-case letter (it's a planet)
    """
    return name[-1].islower() and name[-2] == ' '


def is_candidate(name):
    """
    Check if name ends with a blank + lower-case letter (it's a planet)
    """
    return len(name)>=3 and name[-3] == '.' and name[-2:].isnumeric()


def get_letter(name):
    """
    Extract 'letter' identifier for a planet name.
    Valid confirmed planet names end with a lower-case letter preceded
    by a blank.  Valid planet candidate names end with a dot followed
    by two numbers.

    Examples
    --------
    >>> get_letter('TOI-741.01')
    >>> get_letter('WASP-69 b')
    """
    if is_letter(name):
        return name[-2:]
    if '.' in name:
        idx = name.rfind('.')
        return name[idx:]
    return ''


def get_host(name):
    """
    Extract host name from a given planet name.
    Valid confirmed planet names end with a lower-case letter preceded
    by a blank.  Valid planet candidate names end with a dot followed
    by two numbers.

    Examples
    --------
    >>> get_host('TOI-741.01')
    >>> get_host('WASP-69 b')
    """
    if is_letter(name):
        return name[:-2]
    if '.' in name:
        idx = name.rfind('.')
        return name[:idx]
    return ''


def select_alias(aka, catalogs, default_name=None):
    """
    Search alternative names take first one found in catalogs list.
    """
    for catalog in catalogs:
        for alias in aka:
            if alias.startswith(catalog):
                return alias
    return default_name


def invert_aliases(aliases):
    """
    Invert an {alias:name} dictionary into {name:aliases_list}
    """
    aka = {}
    for key,val in aliases.items():
        if val not in aka:
            aka[val] = []
        aka[val].append(key)
    return aka


def to_float(value):
    """
    Cast string to None or float type.
    """
    if value == 'None':
        return None
    return float(value)


def as_str(val, fmt='', if_none=None):
    """
    Format as string
    """
    if val is None or np.isnan(val):
        return if_none
    return f'{val:{fmt}}'

