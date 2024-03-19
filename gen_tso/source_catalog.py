# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'load_nea_targets_table',
    'load_trexolits_table',
    'normalize_name',
]

import pickle

from astropy.io import ascii
import numpy as np
import requests


# CGS constants:
rsun = 69570000000.0
rearth = 637810000.0
au = 14959787070000.0
G = 6.6743e-08
msun = 1.9885e+33
day = 86400.0


def load_nea_targets_table():
    """
    Unpack star and planet properties from plain text file.

    Examples
    --------
    >>> import source_catalog as cat
    >>> nea_data = cat.load_nea_targets_table()
    """
    with open('../data/nea_data.txt', 'r') as f:
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
            ks_mag.append(to_float(st_mag))
            teff.append(to_float(st_teff))
            log_g.append(to_float(st_logg))
            tr_dur.append(to_float(pl_tr_dur))
            rprs.append(to_float(pl_rprs))
            teq.append(to_float(pl_teq))

    return planets, hosts, ra, dec, ks_mag, teff, log_g, tr_dur, rprs, teq


def load_trexolits_table(all_aliases=False):
    """
    Get the list of targets in trexolists (as named at the NEA).
    A dictionary of name aliases contains alternative names found.

    Examples
    --------
    >>> import source_catalog as cat
    >>> targets, aliases, missing = cat.load_trexolits_table()
    """
    nea_data = load_nea_targets_table()
    hosts = nea_data[1]

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

    # jwst targets that are in nea list:
    jwst_targets = list(norm_targets[np.in1d(norm_targets, hosts)])

    # Missing targets, might be because of name aliases
    if all_aliases:
        with open('../data/nea_all_aliases.pickle', 'rb') as handle:
            aliases = pickle.load(handle)
    else:
        aliases = load_aliases(as_hosts=True)
    alias = {}
    missing = []
    trix = norm_targets[np.in1d(norm_targets, hosts, invert=True)]
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


def load_aliases(as_hosts=False):
    """
    Load file with known aliases of NEA targets.
    """
    with open('../data/nea_aliases.txt', 'r') as f:
        lines = f.readlines()

    def is_letter(name):
        return name[-1].islower() and name[-2] == ' '

    def parse(name):
        if not as_hosts:
            return name
        if is_letter(name):
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


def to_float(value):
    """
    Cast string to None or float type.
    """
    if value == 'None':
        return None
    return float(value)


def fetch_nea_targets_database():
    """
    Fetch (web request) the entire NASA Exoplanet Archive database

    I absolutely need:
    - st_teff (stellar_model, tsm, esm)
    - st_logg (stellar_model)
    - sy_kmag (stellar_model, tsm, esm)
    - pl_trandur (n_integrations, obs_dur)

    I may want to have
    - st_rad (tsm, esm) or [pl_ratror]
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
        "ra,dec,st_teff,st_logg,st_met,st_rad,st_mass,st_age,"
        "pl_trandur,pl_orbper,pl_orbsmax,pl_rade,pl_masse,pl_ratdor,pl_ratror+"
        "from+ps+"
        "&format=json"
    )
    if not r.ok:
        raise ValueError("Something's not OK")

    resp = r.json()
    host_entries = [entry['hostname'] for entry in resp]
    hosts, counts = np.unique(host_entries, return_counts=True)

    planet_entries = np.array([entry['pl_name'] for entry in resp])
    planet_names, idx, counts = np.unique(
        planet_entries,
        return_index=True,
        return_counts=True,
    )
    nplanets = len(planet_names)

    # Make list of unique entries
    planets = [resp[i].copy() for i in idx]
    for i in range(nplanets):
        planet = planets[i]
        name = planet['pl_name']
        idx_duplicates = np.where(planet_entries==name)[0]
        # default_flag takes priority
        def_flags = [resp[j]['default_flag'] for j in idx_duplicates]
        j = idx_duplicates[def_flags.index(1)]
        planets[i] = complete_entry(resp[j].copy())
        dups = [resp[k] for k in idx_duplicates if k!=j]
        rank = rank_planets(dups)
        # Fill the gaps if any
        for j in rank:
            entry = complete_entry(dups[j])
            for field in planet.keys():
                if planets[i][field] is None and entry[field] is not None:
                    planets[i][field] = entry[field]
        planets[i] = complete_entry(planets[i])

    with open('data/nea_data.pickle', 'wb') as handle:
        pickle.dump(planets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # TBD: and make a copy with current date

    def as_str(val, fmt):
        if val is None:
            return None
        return f'{val:{fmt}}'

    # Save as plain text:
    with open('nea_data.txt', 'w') as f:
        host = ''
        for entry in planets:
            ra = entry['ra']
            dec = entry['dec']
            ks_mag = entry['sy_kmag']
            planet = entry['pl_name']
            tr_dur = entry['pl_trandur']
            teff = as_str(entry['st_teff'], '.1f')
            logg = as_str(entry['st_logg'], '.3f')
            rprs = as_str(entry['pl_ratror'], '.3f')
            teq = t_eq(entry['st_teff'], entry['st_rad'], entry['pl_orbsmax'])
            teq = as_str(teq, '.1f')

            if entry['hostname'] != host:
                host = entry['hostname']
                f.write(f">{host}: {ra} {dec} {ks_mag} {teff} {logg}\n")
            f.write(f" {planet}: {tr_dur} {rprs} {teq}\n")


def t_eq(tstar, rstar, sma, f=0.25, A=0.0):
    if tstar is None or rstar is None or sma is None:
        return None
    return tstar * np.sqrt(rstar*rsun/(sma*au)) *(f*(1-A))**0.25


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


def solve_period_sma(period, sma, mstar):
    if mstar is None or mstar == 0:
        return period, sma
    if period is None and sma is not None:
        period = (
            2.0*np.pi * np.sqrt((sma*au)**3.0/G/(mstar*msun)) / day
        )
    elif sma is None and period is not None:
        sma = (
            ((period*day/(2.0*np.pi))**2.0*G*mstar*msun)**(1/3)/au
        )
    return period, sma


def solve_rp_rs(rp, rs, rprs):
    if rp is None and rs is not None and rprs is not None:
        rp = rprs * (rs*rsun) / rearth
    if rs is None and rp is not None and rprs is not None:
        rs = rp*rearth / rprs / rsun
    if rprs is None and rp is not None and rs is not None:
        rprs = rp*rearth / (rs*rsun)
    return rp, rs, rprs

def solve_a_rs(a, rs, ars):
    if a is None and rs is not None and ars is not None:
        a = ars * (rs*rsun) / au
    if rs is None and a is not None and ars is not None:
        rs = a*au / ars / rsun
    if ars is None and a is not None and rs is not None:
        ars = a*au / (rs*rsun)
    return a, rs, ars

