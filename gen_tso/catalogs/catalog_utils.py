# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

"""
A collection of low-level routines to handle catalogs.
"""

__all__ = [
    'normalize_name',
    'get_trexolists_targets',
    'is_letter',
    'is_candidate',
    'get_letter',
    'select_alias',
    'invert_aliases',
    'rank_planets',
    'solve_period_sma',
    'solve_rp_rs',
    'solve_a_rs',
    'complete_entry',
    'to_float',
    'as_str',
]

import re

from astropy.io import ascii
import numpy as np
import pyratbay.constants as pc

from ..utils import ROOT


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
    # Prefixes
    name = name.replace('GL', 'GJ')
    prefixes = [
        'L', 'G', 'HD', 'GJ', 'LTT', 'LHS', 'HIP', 'WD', 'LP', '2MASS', 'PSR',
    ]
    for prefix in prefixes:
        prefix_len = len(prefix)
        if name.startswith(prefix) and not name[prefix_len].isalpha():
            name = name.replace(f'{prefix}-', f'{prefix} ')
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
    # Custom corrections:
    if name in ['55CNC', 'RHO01-CNC']:
        name = '55 Cnc'
    name = name.replace('-offset', '')
    name = name.replace('-updated', '')
    if name.endswith('-'):
        name = name[:-1]
    if name == 'WD 1856':
        name = 'WD 1856+534'
    if 'V1298' in name:
        name = 'V1298 Tau'
    return name


def get_trexolists_targets(grouped=False, trexo_file=None, extract='Target'):
    """
    Get the target names from the trexolists.csv file.

    Parameters
    ----------
    grouped: Bool
        If False, return a 1D list of names.
        If True, return a nested list of lists, with each item the
        set of names for a same object.
    """
    if trexo_file is None:
        trexo_file = f'{ROOT}data/trexolists.csv'

    trexolist_data = ascii.read(
        trexo_file,
        format='csv', guess=False, fast_reader=False, comment='#',
    )

    targets = trexolist_data['Target'].data
    norm_targets = [
        normalize_name(target)
        for target in targets
    ]
    if not grouped:
        return np.unique(norm_targets)

    # Use RA and dec to detect aliases for a same object
    ra = trexolist_data['R.A. 2000'].data
    dec = trexolist_data['Dec. 2000'].data
    truncated_ra = [r[0:7] for r in ra]
    truncated_dec = [d[0:6] for d in dec]

    target_sets = []
    ntargets = len(targets)
    taken = np.zeros(ntargets, bool)
    for i in range(ntargets):
        if taken[i]:
            continue
        ra = truncated_ra[i]
        dec = truncated_dec[i]
        taken[i] = True
        hosts = [norm_targets[i]]
        for j in range(i,ntargets):
            if truncated_ra[j]==ra and truncated_dec[j]==dec and not taken[j]:
                hosts.append(norm_targets[j])
                taken[j] = True
        # print(f'{ra}  {dec}   {np.unique(hosts)}')
        target_sets.append(list(np.unique(hosts)))
    return target_sets


def is_letter(name):
    """
    Check if name ends with a blank + lower-case letter (it's a planet)
    """
    return name[-1].islower() and name[-2] == ' '


def is_candidate(name):
    """
    Check if name ends with a blank + lower-case letter (it's a planet)
    """
    return name[-3] == '.' and name[-2:].isnumeric()


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


def rank_planets(entries):
    """
    Rank entries with the most data
    """
    points = [
        (
            (entry['st_teff'] is None) +
            (entry['st_logg'] is None) +
            (entry['st_met'] is None) +
            (entry['pl_trandur'] is None) +
            (entry['pl_rade'] is None) +
            (entry['pl_orbsmax'] is None and entry['pl_ratdor'] is None) +
            (entry['st_rad'] is None and entry['pl_ratror'] is None)
        )
        for entry in entries
    ]
    rank = np.argsort(np.array(points))
    return rank


def solve_period_sma(period, sma, mstar):
    """
    Solve period-sma-mstar system values.
    """
    if mstar is None or mstar == 0:
        return period, sma
    if period is None and sma is not None:
        period = (
            2.0*np.pi * np.sqrt((sma*pc.au)**3.0/pc.G/(mstar*pc.msun)) / pc.day
        )
    elif sma is None and period is not None:
        sma = (
            ((period*pc.day/(2.0*np.pi))**2.0*pc.G*mstar*pc.msun)**(1/3) / pc.au
        )
    return period, sma


def solve_rp_rs(rp, rs, rprs):
    if rp is None and rs is not None and rprs is not None:
        rp = rprs * (rs*pc.rsun) / pc.rearth
    if rs is None and rp is not None and rprs is not None:
        rs = rp*pc.rearth / rprs / pc.rsun
    if rprs is None and rp is not None and rs is not None:
        rprs = rp*pc.rearth / (rs*pc.rsun)
    return rp, rs, rprs


def solve_a_rs(a, rs, ars):
    if a is None and rs is not None and ars is not None:
        a = ars * (rs*pc.rsun) / pc.au
    if rs is None and a is not None and ars is not None:
        rs = a*pc.au / ars / pc.rsun
    if ars is None and a is not None and rs is not None:
        ars = a*pc.au / (rs*pc.rsun)
    return a, rs, ars


def complete_entry(entry):
    entry['pl_rade'], entry['st_rad'], entry['pl_ratror'] = solve_rp_rs(
        entry['pl_rade'], entry['st_rad'], entry['pl_ratror'],
    )
    entry['pl_orbsmax'], entry['st_rad'], entry['pl_ratdor'] = solve_a_rs(
        entry['pl_orbsmax'], entry['st_rad'], entry['pl_ratdor'],
    )
    entry['pl_orbper'], entry['pl_orbsmax'] = solve_period_sma(
        entry['pl_orbper'], entry['pl_orbsmax'], entry['st_mass']
    )
    return entry


def to_float(value):
    """
    Cast string to None or float type.
    """
    if value == 'None':
        return None
    return float(value)


def as_str(val, fmt):
    """
    Format as string
    """
    if val is None:
        return 'None'
    return f'{val:{fmt}}'

