# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'load_nea_targets_table',
    'load_trexolits_table',
    'normalize_name',
]

import multiprocessing as mp
import pickle
import re
import urllib
import warnings

from astropy.io import ascii
import numpy as np
from astroquery.simbad import Simbad as simbad
import requests


def load_nea_targets_table():
    with open('../data/nea_data.pickle', 'rb') as handle:
        nea = pickle.load(handle)

    planets = [planet['pl_name'] for planet in nea]
    hosts = [planet['hostname'] for planet in nea]
    teff  = [planet['st_teff'] for planet in nea]
    log_g = [planet['st_logg'] for planet in nea]
    ks_mag = [planet['sy_kmag'] for planet in nea]
    tr_dur = [planet['pl_trandur'] for planet in nea]

    return planets, hosts, teff, log_g, ks_mag, tr_dur


def load_nea_tess_table():
    with open('../data/nea_tess_candidates.pickle', 'rb') as handle:
        tess = pickle.load(handle)
    return tess


def load_trexolits_table():
    """
    Get the list of targets in trexolists (as named at the NEA).
    A dictionary of name aliases contains alternative names found.

    >>> targets, aliases, missing = load_trexolits_table()
    """
    nea_data = load_nea_targets_table()
    hosts = nea_data[1]
    with open('../data/nea_all_aliases.pickle', 'rb') as handle:
        aliases = pickle.load(handle)

    trexolist_data = ascii.read(
        '../data/trexolists.csv',
        format='csv', guess=False, fast_reader=False, comment='#',
    )
    targets = np.unique(trexolist_data['Target'].data)

    norm_targets = []
    for target in targets:
        name = normalize_name(target)
        norm_targets.append(name)
    norm_targets = np.unique(norm_targets)

    trix = norm_targets[np.in1d(norm_targets, hosts, invert=True)]
    jwst_targets = list(norm_targets[np.in1d(norm_targets, hosts)])
    alias = {}
    missing = []
    for target in trix:
        if target in aliases:
            alias[target] = aliases[target]
            jwst_targets.append(aliases[target])
        elif target.endswith(' A') and target[:-2] in aliases:
            alias[target] = aliases[target[:-2]]
            jwst_targets.append(aliases[target[:-2]])
        else:
            missing.append(target)

    return np.unique(jwst_targets), alias, np.unique(missing)


def normalize_name(target):
    """
    Normalize target names into a 'more standard' format.
    Mainly to resolve trexolists target names.
    """
    name = target
    # It's a case issue:
    name = name.replace('KEPLER', 'Kepler')
    name = name.replace('TRES', 'TrES')
    name = name.replace('WOLF-', 'Wolf ')
    name = name.replace('HATP', 'HAT-P-')
    # Prefixes
    name = name.replace('GL', 'GJ')
    prefixes = ['L', 'G', 'HD', 'GJ', 'LTT', 'LHS', 'HIP', 'WD', 'LP', '2MASS']
    for prefix in prefixes:
        prefix_len = len(prefix)
        if name.startswith(prefix) and not name[prefix_len].isalpha():
            name = target.replace(f'{prefix}-', f'{prefix} ')
            if name[prefix_len] != ' ':
                name = f'{prefix} ' + name[prefix_len:]
    if name.startswith('CD-'):
        dash_loc = name.index('-', 3)
        name = name[0:dash_loc] + ' ' + name[dash_loc+1:]
    # Main star
    if name.endswith('A'):
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


def fetch_nea_targets_database():
    """
    Fetch (web request) the entire NASA Exoplanet Archive database

    absolutely need:
    - st_teff (stellar_model, tsm, esm)
    - st_logg (stellar_model)
    - sy_kmag (stellar_model, tsm, esm)
    - st_rad (tsm, esm)   or [pl_ratror]
    - pl_rade (tsm, esm)
    - pl_masse (tsm)
    - pl_orbsmax (tsm, esm)  or [pl_ratdor] or [pl_orbper and st_mass]
    """
    # COLUMN pl_name:        Planet Name
    # COLUMN hostname:       Host Name
    # COLUMN default_flag:   Default Parameter Set
    # COLUMN sy_pnum:        Number of Planets
    # COLUMN sy_kmag:        Ks (2MASS) Magnitude

    # COLUMN st_spectype:    Spectral Type
    # COLUMN st_teff:        Stellar Effective Temperature [K]
    # COLUMN st_rad:         Stellar Radius [Solar Radius]
    # COLUMN st_mass:        Stellar Mass [Solar mass]
    # COLUMN st_met:         Stellar Metallicity [dex]
    # COLUMN st_age:         Stellar Age [Gyr]

    # COLUMN pl_orbper:      Orbital Period [days]
    # COLUMN pl_orbsmax:     Orbit Semi-Major Axis [au])
    # COLUMN pl_rade:        Planet Radius [Earth Radius]
    # COLUMN pl_radj:        Planet Radius [Jupiter Radius]
    # COLUMN pl_massj:       Planet Mass [Jupiter Mass]
    # COLUMN pl_eqt:         Equilibrium Temperature [K]
    # COLUMN pl_ratdor:      Ratio of Semi-Major Axis to Stellar Radius
    # COLUMN pl_ratror:      Ratio of Planet to Stellar Radius

    # Fetch all planetary system entries
    r = requests.get(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        "select+hostname,pl_name,default_flag,sy_kmag,sy_pnum,disc_facility,"
        "st_teff,st_logg,st_met,st_rad,st_mass,st_age,"
        "pl_trandur,pl_orbper,pl_orbsmax,pl_rade,pl_masse,pl_ratdor,pl_ratror+"
        "from+ps+"
        "&format=json"
    )
    if not r.ok:
        raise ValueError("Something's not OK")

    resp = r.json()
    host_entries = [entry['hostname'] for entry in resp]
    hosts, counts = np.unique(host_entries, return_counts=True)
    nstars = len(hosts)

    planet_entries = np.array([entry['pl_name'] for entry in resp])
    planet_names, idx, counts = np.unique(
        planet_entries,
        return_index=True,
        return_counts=True,
    )
    nplanets = len(planet_names)
    print(nstars, nplanets)

    # Make list of unique entries (don't worry yet for the howto)
    planets = [resp[i].copy() for i in idx]
    for i in range(nplanets):
        planet = planets[i]
        name = planet['pl_name']
        idx_duplicates = np.where(planet_entries==name)[0]
        def_flags = [resp[j]['default_flag'] for j in idx_duplicates]
        if np.any(def_flags):
            j = idx_duplicates[def_flags.index(1)]
            planets[i] = resp[j].copy()
        for j in idx_duplicates:
            for field in planet.keys():
                if planets[i][field] is None and resp[j][field] is not None:
                    #print(name, field, resp[j][field])
                    planets[i][field] = resp[j][field]

    transiting = [planet['pl_trandur'] is not None for planet in planets]
    ntransit = np.sum(transiting)
    print(nstars, nplanets, ntransit)

    with open('data/nea_data.pickle', 'wb') as handle:
        pickle.dump(planets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # TBD: and make a copy with current date


def fetch_simbad_aliases(target, verbose=True):
    """
    Fetch target aliases as known by Simbad.
    Also get the target Ks magnitude.
    """
    simbad.reset_votable_fields()
    simbad.remove_votable_fields('coordinates')
    simbad.add_votable_fields("otype", "otypes", "ids")
    simbad.add_votable_fields("flux(K)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simbad_info = simbad.query_object(target)
    if simbad_info is None:
        if verbose:
            print(f'Simbad target {repr(target)} not found')
        return [], None

    object_type = simbad_info['OTYPE'].value.data[0]
    if 'Planet' in object_type:
        if target[-1].isalpha():
            host = target[:-1]
        elif '.' in target:
            end = target.rindex('.')
            host = target[:end]
        else:
            target_id = simbad_info['MAIN_ID'].value.data[0]
            print(f'Wait, what?:  {repr(target)}  {repr(target_id)}')
            return [], None
        # go after star
        simbad_info = simbad.query_object(host)
        if simbad_info is None:
            if verbose:
                print(f'Simbad host {repr(host)} not found')
            return [], None

    host_info = simbad_info['IDS'].value.data[0]
    host_alias = host_info.split('|')
    kmag = simbad_info['FLUX_K'].value.data[0]

    return host_alias, kmag


def fetch_nea_aliases(target):
    """
    Fetch target aliases as known by https://exoplanetarchive.ipac.caltech.edu/
    This one is quite slow, it would be great if one could do a batch search.
    """
    query = urllib.parse.quote(target)
    r = requests.get(
        'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Lookup/'
        f'nph-aliaslookup.py?objname={query}'
    )
    resp = r.json()

    aliases = {}
    star_set = resp['system']['objects']['stellar_set']['stars']
    for star in star_set.keys():
        if 'is_host' not in star_set[star]:
            continue
        for alias in star_set[star]['alias_set']['aliases']:
            aliases[alias] = star
        # Do not fetch Simbad aliases here because too many requests
        # break the code

    planet_set = resp['system']['objects']['planet_set']['planets']
    for planet in planet_set.keys():
        for alias in planet_set[planet]['alias_set']['aliases']:
            aliases[alias] = planet
    return aliases


def fetch_all_aliases():
    """
    Yes, all of them, one by one. This will take a while to run.
    """
    nea_data = load_nea_targets_table()
    tess = load_nea_tess_table()
    hosts = np.unique(nea_data[1])
    nhosts = len(hosts)

    aliases = {}
    chunksize = 24
    nchunks = nhosts // chunksize
    k = 0
    for k in range(k, nchunks+1):
        first = k*chunksize
        last = np.clip((k+1) * chunksize, 0, nhosts)
        with mp.get_context('fork').Pool(8) as pool:
            new_aliases = pool.map(fetch_nea_aliases, hosts[first:last])

        for new_alias in new_aliases:
            aliases.update(new_alias)
        print(f'{last} / {nhosts}')

    with open('data/nea_all_aliases.pickle', 'wb') as handle:
        pickle.dump(aliases, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Add TESS candidates
    candidates = []
    for candidate in tess:
        keeper = (
            candidate['tfop'] in ['PC', 'KP', 'APC']
            #or (candidate['tfop']=='' and candidate['sy_kmag'] is not None)
        )
        if keeper:
            candidates.append(candidate)
    tess_hosts = [target['hostname'] for target in candidates]
    toi_hosts = [
        host for host in tess_hosts
        if host not in aliases
    ]
    nhosts = len(toi_hosts)
    chunksize = 24
    nchunks = nhosts // chunksize
    k = 0
    for k in range(k, nchunks+1):
        first = k*chunksize
        last = np.clip((k+1) * chunksize, 0, nhosts)
        with mp.get_context('fork').Pool(8) as pool:
            new_aliases = pool.map(fetch_nea_aliases, toi_hosts[first:last])

        for new_alias in new_aliases:
            aliases.update(new_alias)
        print(f'{last} / {nhosts}')

    with open('data/nea_all_aliases.pickle', 'wb') as handle:
        pickle.dump(aliases, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #with open('data/nea_all_aliases.pickle', 'rb') as handle:
    #    aliases = pickle.load(handle)

    # Now contrast against Simbad aliases
    hosts = np.unique(list(nea_data[1]) + toi_hosts)
    nea_names = np.unique(list(aliases.values()))
    for target in hosts:
        s_aliases, kmag = fetch_simbad_aliases(target)
        new_aliases = []
        for alias in s_aliases:
            alias = re.sub(r'\s+', ' ', alias)
            is_new = (
                alias.startswith('G ') or
                alias.startswith('GJ ') or
                alias.startswith('CD-') or
                alias.startswith('Wolf ')
            )
            if is_new and alias not in aliases:
                new_aliases.append(alias)
        if len(new_aliases) == 0:
            continue
        print(f'Target {repr(target)} has new aliases:  {new_aliases}')
        # Add the star aliases
        for alias in new_aliases:
            aliases[alias] = target
            print(f'    {repr(alias)}: {repr(target)}')

            # Add the planet aliases
            for name in nea_names:
                is_child = (
                    name not in hosts and 
                    name.startswith(target) and
                    len(name) > len(target) and
                    name[len(target)] in ['.', ' ']
                )
                if is_child:
                    letter = name[len(target):]
                    aliases[alias+letter] = name
                    print(f'    {repr(alias+letter)}: {repr(name)}')

    with open('data/nea_all_aliases.pickle', 'wb') as handle:
        pickle.dump(aliases, handle, protocol=pickle.HIGHEST_PROTOCOL)

