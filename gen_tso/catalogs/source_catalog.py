# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'load_targets_table',
    'load_trexolist_table',
    'load_aliases',
]

import pickle

from astropy.io import ascii
import numpy as np

from ..utils import ROOT
from . import catalog_utils as u


def load_targets_table(database='nea_data.txt'):
    """
    Unpack star and planet properties from plain text file.

    Parameters
    ----------
    databases: String
        nea_data.txt or tess_data.txt

    Returns
    -------
    planets: 1D string array
    hosts: 1D string array
    ra: 1D float array
    dec: 1D float array
    ks_mag: 1D float array
    teff: 1D float array
    log_g: 1D float array
    tr_dur: 1D float array
    rprs: 1D float array
    teq: 1D float array

    Examples
    --------
    >>> import source_catalog as cat
    >>> nea_data = cat.load_nea_targets_table()
    """
    with open(f'{ROOT}data/{database}', 'r') as f:
        lines = f.readlines()

    planets = []
    hosts = []
    ra = []
    dec = []
    ks_mag = []
    teff = []
    log_g = []
    tr_dur = []
    rprs = []
    teq = []

    for line in lines:
        if line.startswith('>'):
            name_len = line.find(':')
            host = line[1:name_len]
            st_ra, st_dec, st_mag, st_teff, st_logg = line[name_len+1:].split()
        elif line.startswith(' '):
            name_len = line.find(':')
            planet = line[1:name_len].strip()
            pl_tr_dur, pl_rprs, pl_teq = line[name_len+1:].split()

            planets.append(planet)
            hosts.append(host)
            ra.append(float(st_ra))
            dec.append(float(st_dec))
            ks_mag.append(u.to_float(st_mag))
            teff.append(u.to_float(st_teff))
            log_g.append(u.to_float(st_logg))
            tr_dur.append(u.to_float(pl_tr_dur))
            rprs.append(u.to_float(pl_rprs))
            teq.append(u.to_float(pl_teq))

    return planets, hosts, ra, dec, ks_mag, teff, log_g, tr_dur, rprs, teq


def load_trexolist_table(all_aliases=False):
    """
    Get the list of targets in trexolists (as named at the NEA).
    A dictionary of name aliases contains alternative names found.

    Returns
    -------
    jwst_targets: List
        trexolists host names as found in the NEA database.
    aliases: Dict
        aliases of hosts as found in the trexolists database.
    missing: List
        trexolists hosts not found in the NEA database.
    original_names: Dict
        Names of targets as listed in the trexolists database.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> targets, aliases, missing, og = cat.load_trexolist_table(True)
    """
    trexolist_data = ascii.read(
        f'{ROOT}data/trexolists.csv',
        format='csv', guess=False, fast_reader=False, comment='#',
    )
    targets = np.unique(trexolist_data['Target'].data)

    original_names = {}
    norm_targets = []
    for target in targets:
        name = u.normalize_name(target)
        norm_targets.append(name)
        if name not in original_names:
            original_names[name] = []
        original_names[name] += [target]
    norm_targets = np.unique(norm_targets)

    # jwst targets that are in NEA catalog:
    nea_data = load_targets_table('nea_data.txt')
    tess_data = load_targets_table('tess_data.txt')
    hosts = list(nea_data[1]) + list(tess_data[1])

    if all_aliases:
        with open(f'{ROOT}data/nea_aliases.pickle', 'rb') as handle:
            aliases = pickle.load(handle)
        with open(f'{ROOT}data/tess_aliases.pickle', 'rb') as handle:
            tess_aliases = pickle.load(handle)
        aliases.update(tess_aliases)
    else:
        aliases = load_aliases(as_hosts=True)
        for host in hosts:
            aliases[host] = host

    # As named in NEA catalogs:
    jwst_aliases = {}
    missing = []
    for target in norm_targets:
        if target in aliases:
            jwst_aliases[target] = aliases[target]
        elif target.endswith(' A') and target[:-2] in aliases:
            jwst_aliases[target] = aliases[target[:-2]]
        else:
            missing.append(target)
    for name in list(jwst_aliases.values()):
        jwst_aliases[name] = name

    # TBD: Check this does not break for incomplete lists
    trexo_names = {}
    for name in original_names:
        alias = jwst_aliases[name]
        if alias not in trexo_names:
            trexo_names[alias] = []
        trexo_names[alias] += original_names[name]

    jwst_targets = np.unique(list(jwst_aliases.values()))
    return jwst_targets, jwst_aliases, np.unique(missing), trexo_names


def load_aliases(as_hosts=False):
    """
    Load file with known aliases of NEA targets.
    """
    with open(f'{ROOT}data/nea_aliases.txt', 'r') as f:
        lines = f.readlines()

    def parse(name):
        if not as_hosts:
            return name
        if u.is_letter(name):
            return name[:-2]
        end = name.rindex('.')
        return name[:end]

    aliases = {}
    for line in lines:
        loc = line.index(':')
        name = parse(line[:loc])
        for alias in line[loc+1:].strip().split(','):
            aliases[parse(alias)] = name
    return aliases

