# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'fetch_trexolist',
    'update_exoplanet_archive',
    'fetch_nasa_confirmed_targets',
    'fetch_gaia_targets',
    'fetch_nea_aliases',
    '_load_jwst_names',
]


import concurrent.futures
import multiprocessing as mp
from datetime import datetime, timezone
import os
import pickle
import re
import socket
import ssl
import urllib
import warnings


from astropy.coordinates import SkyCoord
import numpy as np
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astropy.table import Table
from astropy.units import arcsec, deg
from bs4 import BeautifulSoup
import pyratbay.constants as pc
import requests

from ..utils import ROOT
from .catalogs import (
    load_targets, load_trexolists, load_programs, load_aliases,
    _group_by_target,
)
from . import utils as u
from . import target as tar
from .target import Target


def format_nea_entry(entry):
    """
    Have TOI entries the same keys as PS entries.
    Turn None values into np.nan.
    Calculate stellar mass from logg-rstar for TESS candidates
    """
    if 'toi' in entry.keys():
        entry['hostname'] = f"TOI-{entry['toipfx']}"
        entry['st_met'] = np.nan
        entry['sy_kmag'] = np.nan
        logg = entry['st_logg'] if entry['st_logg'] is not None else np.nan
        rstar = entry['st_rad'] if entry['st_rad'] is not None else np.nan
        entry['st_mass'] = 10**logg * (rstar*pc.rsun)**2 / pc.G / pc.msun

        entry['pl_name'] = f"TOI-{entry['toi']}"
        entry['pl_masse'] = np.nan
        entry['pl_orbsmax'] = np.nan
        entry['pl_ratdor'] = np.nan
        entry['pl_ratror'] = np.sqrt(entry.pop('pl_trandep')*pc.ppm)
        entry['pl_trandur'] = entry.pop('pl_trandurh')
        entry['pl_msinie'] = np.nan
    else:
        entry['pl_eqt'] = np.nan

    # Patch
    if entry['st_mass'] == 0.0:
        entry['st_mass'] = np.nan

    # Replace None with np.nan
    for key in entry.keys():
        if entry[key] is None:
            entry[key] = np.nan
    return entry


def get_children(host_aliases, planet_aliases):
    """
    Cross check a dictionary of star and planet aliases to see
    whether the star is the host of the planets.
    """
    # get all planet aliases minus the 'letter' identifier
    planet_aka = u.invert_aliases(planet_aliases)
    for planet, aliases in planet_aka.items():
        aka = []
        for alias in aliases:
            len_letter = len(u.get_letter(alias))
            aka.append(alias[0:-len_letter])
        planet_aka[planet] = aka

    # cross_check with host aliases
    children = []
    for planet, aliases in planet_aka.items():
        if np.any(np.isin(aliases, host_aliases)):
            children.append(planet)

    aliases = {
        alias:planet
        for alias,planet in planet_aliases.items()
        if planet in children
    }
    return aliases


def save_catalog(targets, catalog_file):
    """
    Write data from a catalog of targets to a plain-text file.
    Targets will be sorted by host name and then by planet name.
    """
    # Save as plain text:
    with open(catalog_file, 'w') as f:
        f.write(
            '# > host: RA(deg) dec(deg) Ks_mag '
            'rstar(rsun) mstar(msun) teff(K) log_g metallicity(dex)\n'
            '# planet: T14(h) rplanet(rearth) mplanet(mearth) '
            'semi-major_axis(AU) period(d) t_eq(K) is_min_mass\n'
        )
        hosts = [target.host for target in targets]
        planets = [target.planet for target in targets]
        isort = np.lexsort((planets, hosts))
        host = ''
        for idx in isort:
            target = targets[idx]
            planet = target.planet
            ra = f'{target.ra:.7f}'
            dec = f'{target.dec:.7f}'
            ks_mag = f'{target.ks_mag:.3f}'
            teff = f'{target.teff:.1f}'
            rstar = f'{target.rstar:.3f}'
            mstar = f'{target.mstar:.3f}'
            logg = f'{target.logg_star:.2f}'
            metal = f'{target.metal_star:.2f}'
            rplanet = f'{target.rplanet:.3f}'
            mplanet = f'{target.mplanet:.3f}'
            transit_dur = f'{target.transit_dur:.3f}'
            sma = f'{target.sma:.4f}'
            period = f'{target.period:.5f}'
            teq = f'{target.eq_temp:.1f}'
            is_min_mass = int(target.is_min_mass)
            if target.host != host:
                host = target.host
                f.write(
                    f">{host}: {ra} {dec} {ks_mag} "
                    f"{rstar} {mstar} {teff} {logg} {metal}\n",
                )
            f.write(
                f" {planet}: {transit_dur} {rplanet} {mplanet} "
                f"{sma} {period} {teq} {is_min_mass}\n",
            )


def update_exoplanet_archive(from_scratch=False):
    """
    Fetch confirmed and TESS-candidate targets from the NASA
    Exoplanet archive.

    Parameters
    ----------
    from_scratch: Bool
        If True, fetch all aliases from scratch.  The run will
        take much longer but it will populate the local .pickle files
        with all known aliases, making later runs more efficient.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> cat.update_exoplanet_archive()
    """
    # Update trexolist database
    fetch_trexolist()

    # NEA confirmed targets
    print('\nFetching confirmed planets from the NASA archive')
    new_targets = fetch_nasa_confirmed_targets()
    if from_scratch:
        new_targets = None
    print('Fetching confirmed planets aliases')
    fetch_confirmed_aliases(new_targets)

    # NEA TESS candidate targets
    print('Fetching TESS candidate planets from the NASA archive')
    new_targets = fetch_nasa_tess_candidates()
    if from_scratch:
        new_targets = None
    print('Fetching TESS candidates aliases')
    fetch_tess_aliases(new_targets)
    crosscheck_tess_candidates()

    today = datetime.now(timezone.utc)
    with open(f'{ROOT}/data/last_updated_nea.txt', 'w') as f:
        f.write(f'{today.year}_{today.month:02}_{today.day:02}')

    # Update aliases list
    curate_aliases()
    print('Exoplanet data is up to date')


def _load_jwst_names(grouped=False):
    """
    Keep track of JWST target aliases from programs to cross-check
    from gen_tso.catalogs import load_trexolists, load_programs
    from gen_tso.catalogs import *
    """
    trexo_data = load_trexolists(grouped=False)
    observations = load_programs(grouped=False)
    if not grouped:
        jwst_names = np.unique([obs['target'] for obs in trexo_data])
        known_targets = np.unique([obs['target'] for obs in observations])
        jwst_names = np.union1d(jwst_names, known_targets).tolist()
        return jwst_names

    # grouped by target
    all_obs = trexo_data + observations
    observations = _group_by_target(all_obs)
    jwst_aliases = []
    for obs_group in observations:
        names = np.unique([obs['target'] for obs in obs_group]).tolist()
        jwst_aliases.append(names)

    return jwst_aliases


def curate_aliases():
    """
    Thin down all_aliases.pickle file to the essentials.
    Save as .txt, which is shipped with gen_tso.
    """
    with open(f'{ROOT}data/nea_aliases.pickle', 'rb') as handle:
        aliases = pickle.load(handle)
    with open(f'{ROOT}data/tess_aliases.pickle', 'rb') as handle:
        tess_aliases = pickle.load(handle)
    aliases.update(tess_aliases)

    # Ensure to match against NEA host names for jwst targets
    jwst_names = _load_jwst_names()
    for host,system in aliases.items():
        is_in = np.isin(system['host_aliases'], jwst_names)
        if np.any(is_in) and system['host'] not in jwst_names:
            jwst_names.append(system['host'])

    prefixes = jwst_names
    prefixes += ['WASP', 'KELT', 'HAT', 'MASCARA', 'TOI', 'XO', 'TrES']
    keep_aliases = {}
    for host,system in aliases.items():
        for alias,planet in system['planet_aliases'].items():
            alias_host = u.get_host(alias)
            for prefix in prefixes:
                if alias.startswith(prefix) and alias != planet:
                    keep_aliases[alias] = planet
            if alias not in keep_aliases and alias_host == host and alias != planet:
                keep_aliases[alias] = planet

    aka = u.invert_aliases(keep_aliases)
    to_remove = []
    for name, aliases in aka.items():
        # Keep lettered aliases
        # Keep candidate aliases if lettered alias does not exist
        lettered = [
            u.get_host(alias)
            for alias in aliases
            if u.is_letter(alias)
        ]
        if u.is_letter(name):
            lettered.append(u.get_host(name))
        aliases = [
            alias
            for alias in aliases
            if u.is_letter(alias) or (u.get_host(alias) not in lettered)
        ]
        aka[name] = aliases

        if len(aliases) == 0:
            to_remove.append(name)

    for name in to_remove:
        aka.pop(name)

    sorted_names = sorted(list(aka))
    with open(f'{ROOT}data/target_aliases.txt', 'w') as f:
        for name in sorted_names:
            # TBD: catch this, programatically
            if name == 'LP 261-75 C':
                continue
            aliases = sorted(aka[name])
            str_aliases = ','.join(aliases)
            f.write(f'{name}:{str_aliases}\n')


def fetch_trexolist():
    """
    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> cat.fetch_trexolist()
    """
    # Fetch the data:
    url = "https://www.stsci.edu/~nnikolov/TrExoLiSTS/JWST/03_trexolists_extended.csv"
    query_parameters = {}
    response = requests.get(url, params=query_parameters)

    if not response.ok:
        raise ValueError('Could not download TrExoLiSTS database')

    trexolists_path = f'{ROOT}data/trexolists.csv'
    with open(trexolists_path, mode="wb") as file:
        file.write(response.content)

    # Fetch the last-update date:
    url = "https://www.stsci.edu/~nnikolov/TrExoLiSTS/JWST/trexolists.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    last_update_tag = soup.find('h3', string=lambda text: 'Last update' in text)
    last_update_text = last_update_tag.get_text(strip=True)
    date = datetime.strptime(last_update_text[13:], '%Y-%m-%d %H:%M:%S')
    with open(f'{ROOT}data/last_updated_trexolist.txt', 'w') as f:
        f.write(f'{date.year}_{date.month:02}_{date.day:02}')


def fetch_nasa_confirmed_targets():
    """
    Fetch (HTTP web request) the entire NASA Exoplanet Archive database
    for confirmed planets (there is another one for TESS candidates)

    Returns
    -------
    new_targets: List of strings
        List of target names flagged as updated by the NEA
        since the last update (rowupdate column).

    See also
    --------
    - fetch_nasa_tess_candidates()  to fetch the TESS candidates database

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> new_targets = cat.fetch_nasa_confirmed_targets()
    """
    # Fetch all planetary system entries
    r = requests.get(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        "select+hostname,pl_name,default_flag,rowupdate,sy_kmag,sy_pnum,"
        "ra,dec,st_teff,st_logg,st_met,st_rad,st_mass,st_age,pl_trandur,"
        "pl_orbper,pl_orbsmax,pl_rade,pl_masse,pl_msinie,pl_ratdor,pl_ratror+"
        "from+ps+"
        "&format=json"
    )
    if not r.ok:
        raise ValueError("Something's not OK")
    resp = [format_nea_entry(entry) for entry in r.json()]

    hosts = np.unique([entry['hostname'] for entry in resp])
    default_flags = np.array([entry['default_flag'] for entry in resp])
    planet_entries = np.array([entry['pl_name'] for entry in resp])
    planet_names, idx, counts = np.unique(
        planet_entries,
        return_index=True,
        return_counts=True,
    )

    targets = []
    # Make list of unique entries
    # Group by host such that planets share same host-star values
    for host in hosts:
        children, counts = np.unique(
            [entry['pl_name'] for entry in resp if entry['hostname']==host],
            return_counts=True,
        )
        planets = []
        n_dups = []
        for name in children:
            idx_entry = np.where(planet_entries==name)[0]
            entries = [Target(resp[i]) for i in idx_entry]
            j = np.where(default_flags[idx_entry])[0][0]
            target = entries.pop(j)
            tar.rank_planets(target, entries)
            planets.append(target)
            n_dups.append(len(idx_entry))
        # Solve stellar parameters (all planets must have the 'same' host)
        star = tar.solve_host(planets, n_dups)

        # Now, re-do each planet, but using the single host properties
        for name in children:
            idx_entry = np.where(planet_entries==name)[0]
            entries = [Target(resp[i]) for i in idx_entry]
            # Update with star props
            for i in range(len(entries)):
                entries[i].copy_star(star)
            j = np.where(default_flags[idx_entry])[0][0]
            target = entries.pop(j)
            tar.rank_planets(target, entries)
            target._update_dates = [resp[i]['rowupdate'] for i in idx_entry]
            targets.append(target)

    # Find new and updated targets:
    catalog_file = f'{ROOT}data/nea_data.txt'
    if os.path.exists(catalog_file):
        current_targets = [
            target.planet for target in load_targets('nea_data.txt')
        ]
    else:
        current_targets = []

    update_file = f'{ROOT}/data/last_updated_nea.txt'
    if os.path.exists(update_file):
        with open(update_file, 'r') as f:
            date = f.readline().strip()
            last_nasa = datetime.strptime(date,'%Y_%m_%d')
    else:
        last_nasa = datetime.strptime('1990', '%Y')

    new_targets = []
    n_new, n_updated = 0, 0
    for target in targets:
        if target.planet not in current_targets:
            new_targets.append(target.planet)
            n_new += 1
        else:
            is_new = [
                datetime.strptime(date, '%Y-%m-%d') > last_nasa
                for date in target._update_dates
                if isinstance(date, str)
            ]
            if np.any(is_new):
                n_updated += 1
                new_targets.append(target.planet)
    print(
        f'There are {n_new} new and {n_updated} updated confirmed '
        f'targets since {last_nasa.strftime("%Y-%m-%d")}'
    )

    # Save outputs
    save_catalog(targets, catalog_file)
    return new_targets


def fetch_nasa_tess_candidates():
    """
    Fetch entries in the NEA TESS candidates table.
    Remove already confirmed targets.

    Returns
    -------
    new_targets: List of strings
        Targets flagged by NEA to be updated since the last fetch.

    Examples
    --------
    >>> from gen_tso.catalogs import fetch_catalogs as fetch_cat
    >>> new_targets = fetch_cat.fetch_nasa_tess_candidates()
    >>> fetch_cat.fetch_tess_aliases(new_targets)
    >>> fetch_cat.crosscheck_tess_candidates()
    """
    r = requests.get(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        "select+toi,toipfx,pl_trandurh,pl_trandep,pl_rade,pl_eqt,ra,dec,"
        "st_tmag,st_teff,st_logg,st_rad,pl_orbper,tfopwg_disp,rowupdate+"
        "from+toi+"
        "&format=json"
    )
    if not r.ok:
        raise ValueError("Something's not OK")

    entries = [format_nea_entry(entry) for entry in r.json()]
    ntess = len(entries)
    status = [entry['tfopwg_disp'] for entry in entries]
    dates = [entry['rowupdate'][0:10] for entry in entries]

    # Discard confirmed planets:
    targets = load_targets('nea_data.txt')
    confirmed_targets = [target.planet for target in targets]
    confirmed_hosts = [target.host for target in targets]

    # Use multiplicity to vet known targets:
    hosts = [entry['toipfx'] for entry in entries]
    u_hosts, counts = np.unique(hosts, return_counts=True)
    u_hosts = list(u_hosts)
    multiplicity = [
        counts[u_hosts.index(entry['toipfx'])] for entry in entries
    ]
    u_hosts, counts = np.unique(confirmed_hosts, return_counts=True)
    u_hosts = list(u_hosts)
    known_multiplicity = [
        counts[u_hosts.index(target.host)] for target in targets
    ]

    # Get aliases of confirmed planets:
    planet_aliases = {}
    host_aliases = {}

    aliases_file = f'{ROOT}data/target_aliases.txt'
    if os.path.exists(aliases_file):
        known_aliases = load_aliases('system')
        for host, system in known_aliases.items():
            if host in confirmed_hosts:
                planet_aliases.update(system['planet_aliases'])
                for alias in system['host_aliases']:
                    host_aliases[alias] = host

    aliases_file = f'{ROOT}data/nea_aliases.pickle'
    if os.path.exists(aliases_file):
        with open(aliases_file, 'rb') as handle:
            known_aliases = pickle.load(handle)
        for host, system in known_aliases.items():
            planet_aliases.update(system['planet_aliases'])
            for alias in system['host_aliases']:
                host_aliases[alias] = host

    j, k, l = 0, 0, 0
    tess_targets = []
    last_updated = []
    for i in range(ntess):
        target = Target(entries[i])
        # Update names if possible:
        if target.host in host_aliases:
            target.host = host_aliases[target.host]
        if target.planet in planet_aliases:
            target.planet = planet_aliases[target.planet]

        if target.planet in confirmed_targets:
            j += 1
            continue
        if target.host in confirmed_hosts:
            idx = confirmed_hosts.index(target.host)
            if multiplicity[i] == known_multiplicity[idx]:
                continue
            # Update with star props
            k += 1
            target.copy_star(targets[idx])
        if status[i] in ['FA', 'FP']:
            l += 1
            continue
        tess_targets.append(target)
        last_updated.append(datetime.strptime(dates[i], '%Y-%m-%d'))

    # Save temporary data (still need to hunt for Ks mags):
    catalog_file = f'{ROOT}data/tess_candidates_tmp.txt'
    save_catalog(tess_targets, catalog_file)

    with open(f'{ROOT}/data/last_updated_nea.txt', 'r') as f:
        last_nasa = datetime.strptime(f.readline().strip(),'%Y_%m_%d')
    new_targets = [
        target.planet
        for target,last in zip(tess_targets,last_updated)
        if last > last_nasa or target.host in confirmed_hosts
    ]
    return new_targets


def fetch_nea_aliases(targets):
    """
    Fetch target aliases as known by https://exoplanetarchive.ipac.caltech.edu/

    Note 1: a search of a planet or stellar target returns the
        aliases for all bodies in that planetary system.
    Note 2: there might be more than one star per system

    Parameters
    ----------
    targets: String or 1D iterable of strings
        Target(s) to fetch from the NEA database.

    Returns
    -------
    host_aliases_list: 1D list of dictionaries
        List of host-star aliases for each target.
    planet_aliases_list: 1D list of dictionaries
        List of planetary aliases for each target.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> targets = ['WASP-8 b', 'KELT-7', 'HD 189733']
    >>> host_aliases, planet_aliases = cat.fetch_nea_aliases(targets)

    >>> host_aliases, planet_aliases = cat.fetch_nea_aliases('WASP-00')
    """
    if isinstance(targets, str):
        targets = [targets]
    ntargets = len(targets)

    urls = np.array([
        'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Lookup/'
        f'nph-aliaslookup.py?objname={urllib.parse.quote(target)}'
        for target in targets
    ])

    def fetch_url(url):
        try:
            response = requests.get(url)
            return response
        except:
            return None

    fetch_status = np.tile(2, ntargets)
    responses = np.tile({}, ntargets)
    n_attempts = 0
    while np.any(fetch_status>0) and n_attempts < 10:
        n_attempts += 1
        mask = fetch_status > 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_url, urls[mask]))

        j = 0
        for i in range(ntargets):
            if fetch_status[i] <= 0:
                continue
            r = results[j]
            j += 1
            if r is None:
                continue
            if not r.ok:
                warnings.warn(f"Alias fetching failed for '{targets[i]}'")
                fetch_status[i] -= 1
                continue
            responses[i] = r.json()
            fetch_status[i] = 0
        fetched = np.sum(fetch_status <= 0)
        print(f'Fetched {fetched}/{ntargets} entries on try {n_attempts}')

    host_aliases_list = []
    planet_aliases_list = []
    for i,resp in enumerate(responses):
        if resp == {}:
            print(f"NEA alias fetching failed for '{targets[i]}'")
            host_aliases_list.append({})
            planet_aliases_list.append({})
            continue
        if resp['manifest']['lookup_status'] == 'System Not Found':
            print(f"NEA alias not found for '{targets[i]}'")
            host_aliases_list.append({})
            planet_aliases_list.append({})
            continue

        host_aliases = {}
        star_set = resp['system']['objects']['stellar_set']['stars']
        for star in star_set.keys():
            if 'is_host' not in star_set[star]:
                continue
            for alias in star_set[star]['alias_set']['aliases']:
                host_aliases[alias] = star
        host_aliases_list.append(host_aliases)

        planet_aliases = {}
        planet_set = resp['system']['objects']['planet_set']['planets']
        for planet in planet_set.keys():
            for alias in planet_set[planet]['alias_set']['aliases']:
                planet_aliases[alias] = planet
        planet_aliases_list.append(planet_aliases)

    return host_aliases_list, planet_aliases_list


def fetch_simbad_aliases(targets):
    """
    Fetch target aliases and Ks-band magnitude from the Simbad database.

    Parameters
    ----------
    targets: string or iterable of strings
        Name or list of names of astronomical targets to query in SIMBAD.

    Returns
    -------
    aliases: list of lists of str
        A list containing the Simbad aliases (list) for each target.
        If a target is not found, the corresponding list is empty.
    ks_mag: 1D float array
        Ks-band magnitude for each targets.
        If a target is not found, the corresponding magnitude is a np.nan.

    Examples
    --------
    >>> from gen_tso.catalogs.update_catalogs import fetch_simbad_aliases
    >>> # Single-planet search
    >>> aliases, ks_mag = fetch_simbad_aliases('WASP-80')
    >>> print(aliases[0], ks_mag[0], sep='\n')
    ['Gaia DR3 4223507222112425344', 'StKM 2-1435', ..., 'NAME Petra']
    8.35099983215332

    >>> # Batch search
    >>> targets = ['Kepler-138', 'TOI-270', 'WASP-39']
    >>> aliases, ks_mag = fetch_simbad_aliases(targets)
    >>> print(ks_mag)
    [ 9.50599957  8.2510004  10.20199966]

    >>> # Not-found/non-existing targets returns
    >>> aliases, ks_mag = fetch_simbad_aliases('WASP-0')
    >>> print(aliases[0], ks_mag[0], sep='\n')
    []
    nan
    """
    if isinstance(targets, str):
        targets = [targets]
    Simbad.reset_votable_fields()
    Simbad.add_votable_fields("otype", "ids", 'K')

    query = Simbad.query_objects(targets)
    kmag = np.array(query['K'])

    host_aliases = []
    for host in query:
        aliases = host['ids'].split('|')
        # Object not found:
        if host['otype'] == '':
            host_aliases.append([])
            continue
        # Symbad thinks that some hosts are planets:
        if host['otype'] == 'Pl':
            aliases = []
            for alias in host['ids'].split('|'):
                if u.is_candidate(alias) or u.is_letter(alias):
                    alias = u.get_host(alias)
                elif alias[-1] == 'b':
                    alias = alias[:-1]
                aliases.append(alias)
        host_aliases.append(aliases)

    return host_aliases, kmag


def fetch_vizier_ks(target, verbose=True):
    """
    Query for a target in the 2MASS catalog via Vizier.

    Returns
    -------
    Ks_mag: Float
        The target's Ks magnitude.
        Return None if the target was not found in the catalog or
        could not be uniquely identified.

    Examples
    --------
    >>> fetch_vizier_ks('TOI-6927')
    >>> fetch_vizier_ks('2MASS J08024565+2139348')
    >>> fetch_vizier_ks('Gaia DR2 671023360793596672')
    """
    # 2mass catalog
    catalog = 'II/246/out'
    vizier = Vizier(
        catalog=catalog,
        columns=['RAJ2000', 'DEJ2000', '2MASS', 'Kmag'],
        keywords=['Stars'],
    )

    result = vizier.query_object(target, radius=0.5*arcsec)
    n_entries = np.size(result)
    if n_entries == 0:
        print(f"Target not found: '{target}'")
        return np.nan

    data = result[catalog].as_array().data
    if n_entries == 1:
        return data[0][3]
    elif n_entries > 1 and target.startswith('2MASS'):
        # find by name
       for row in data:
           if row[2] == target[-16:]:
               return row[3]
    elif n_entries > 1:
        print(f"Target could not be uniquely identified: '{target}'")
        return np.nan
    return np.nan


def fetch_aliases(hosts, output_file=None, known_aliases=None):
    """
    Fetch aliases from the NEA and Simbad databases for a list
    of host stars.  Store output dictionary of aliases to pickle file.

    Parameters
    ----------
    hosts: List of strings
        Host star names of targets to search.
    output_file: String
        If not None, save outputs to file (as pickle binary format).
    known_aliases: Dictionary
        Dictionary of known aliases, the new aliases will be added
        on top o this dictionary.

    Returns
    -------
    aliases: Dictionary
        Dictionary of aliases with one entry per system where
        the key is the host name.  Each entry is a dictionary
        containing:
            'host' the host name (string)
            'planets': array of planets in the system
            'host_aliases': list of host aliases
            'planet_aliases': dictionary of planets' aliases with
                (key,value) as (alias_name, name)

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> from gen_tso.utils import ROOT

    >>> # Confirmed targets
    >>> targets = cat.load_targets()
    >>> hosts = np.unique([target.host for target in targets])
    >>> output_file = f'{ROOT}data/nea_aliases.pickle'
    >>> aliases = cat.fetch_aliases(hosts, output_file)
    """
    if known_aliases is None:
        known_aliases = {}

    host_aliases, planet_aliases = fetch_nea_aliases(hosts)

    # Ensure to match against JWST program target names
    jwst_aliases = _load_jwst_names(grouped=True)
    jwst_names = np.concatenate(jwst_aliases)

    # Ensure I have the NEA-named host name:
    nhosts = len(hosts)
    host_names = np.array(hosts).tolist()
    for i in range(nhosts):
        if len(host_aliases[i]) == 0:
            continue
        hosts_aka = u.invert_aliases(host_aliases[i])
        for nea_host, h_aliases in hosts_aka.items():
            if hosts[i] in h_aliases:
                host_names[i] = nea_host
                break

    # Get aliases from Simbad:
    simbad_aliases, simbad_kmags = fetch_simbad_aliases(host_names)

    aliases = {}
    nhosts = len(hosts)
    for i in range(nhosts):
        host_name = host_names[i]
        if len(host_aliases[i]) == 0:
            continue
        # Isolate host-planet(s) aliases
        h_aliases = [
            alias
            for alias,host in host_aliases[i].items()
            if host == host_names[i]
        ]
        stars = np.unique(list(host_aliases[i].values()))
        if len(stars) == 1:
            p_aliases = planet_aliases[i].copy()
        else:
            p_aliases = get_children(h_aliases, planet_aliases[i])
        p_aliases = {
            re.sub(r'\s+', ' ', key): val
            for key,val in p_aliases.items()
            if u.is_letter(key) or u.is_candidate(key)
        }
        children_names = np.unique(list(p_aliases.values()))

        in_jwst = np.isin(h_aliases, jwst_names)
        if np.any(in_jwst):
            j_alias = np.array(h_aliases)[in_jwst]

        # Complement with Simbad aliases:
        new_aliases = []
        for alias in simbad_aliases[i]:
            alias = re.sub(r'\s+', ' ', alias)
            is_new = (
                alias in jwst_names or
                alias.startswith('G ') or
                alias.startswith('GJ ') or
                alias.startswith('Wolf ') or
                alias.startswith('2MASS ')
            )
            if is_new and alias not in h_aliases:
                new_aliases.append(alias)
                h_aliases.append(alias)

        # Add JWST host aliases:
        in_jwst = np.isin(h_aliases, jwst_names)
        if np.any(in_jwst):
            j_alias = np.array(h_aliases)[in_jwst][0]
            for j_aliases in jwst_aliases:
                if j_alias not in j_aliases:
                    continue
                h_aliases += [
                    alias
                    for alias in j_aliases
                    if alias not in h_aliases
                ]

        # Replicate host aliases as planet aliases:
        planet_aka = u.invert_aliases(p_aliases)
        for planet, pals in planet_aka.items():
            for host in h_aliases:
                letter = u.get_letter(planet)
                planet_name = f'{host}{letter}'
                # The conditions to add a target:
                is_new = planet_name not in pals
                # There is a planet or a candidate in list
                planet_exists = np.any([
                    u.get_host(p) == host and u.is_letter(p)
                    for p in pals
                ])
                candidate_exists = np.any([
                    u.get_host(p) == host and u.is_candidate(p)
                    for p in pals
                ])
                # Do not downgrade planet -> candidate
                not_downgrade = not (
                    u.is_candidate(planet_name) and
                    planet_exists
                )
                # No previous alias (hold-off TESS names)
                new_entry = (
                    not planet_exists and
                    not candidate_exists and
                    not planet_name.startswith('TOI')
                )
                # There is a letter version of it with same root
                letter_exists = np.any([
                    p.startswith(host) and u.is_letter(p)
                    for p in pals
                ])
                # Upgrade candidate->planet only if is lettered anywhere else
                upgrade = (
                    u.is_letter(planet_name) and
                    candidate_exists and
                    letter_exists
                )
                if is_new and not_downgrade and (new_entry or upgrade):
                    p_aliases[planet_name] = planet

        system = {
            'host': host_name,
            'planets': children_names,
            'host_aliases': h_aliases,
            'planet_aliases': p_aliases,
        }
        aliases[host_name] = system


    # Add previously known aliases (but give priority to the new ones)
    for host in list(known_aliases):
        if host not in aliases:
            aliases[host] = known_aliases[host]

    if output_file is not None:
        with open(output_file, 'wb') as handle:
            pickle.dump(aliases, handle, protocol=4)

    return aliases


def fetch_confirmed_aliases(new_targets=None):
    """
    Fetch aliases for NEA confirmed planets.
    Save results to a binary file 'nea_aliases.pickle'

    Parameters
    ----------
    new_targets: List of strings
        If not None, only fetch aliases for the given targets.


    Examples
    --------
    >>> from gen_tso.catalogs import fetch_catalogs as fetch_cat
    >>> new_targets = fetch_cat.fetch_nasa_confirmed_targets()
    >>> aliases = fetch_cat.fetch_confirmed_aliases(new_targets)
    """
    known_targets = load_targets()
    # Search aliases by host star
    if new_targets is None:
        hosts = np.unique([target.host for target in known_targets])
    else:
        hosts = np.unique([
            target.host for target in known_targets
            if target.planet in new_targets
        ])

    # Get previously known aliases
    known_aliases = {}
    aliases_file = f'{ROOT}data/target_aliases.txt'
    if os.path.exists(aliases_file):
        known_aliases = load_aliases('system')
    output_file = f'{ROOT}data/nea_aliases.pickle'
    if os.path.exists(output_file):
        with open(output_file, 'rb') as handle:
            prev_aliases = pickle.load(handle)
        for host,system in prev_aliases.items():
            known_aliases[host] = system

    known_hosts = np.unique([target.host for target in known_targets])
    for host in list(known_aliases):
        if host not in known_hosts:
            known_aliases.pop(host)

    # Get new aliases
    aliases = fetch_aliases(hosts, output_file, known_aliases)
    return aliases


def fetch_tess_aliases(new_targets=None):
    """
    Get TESS candidate aliases.
    You want to run fetch_nasa_tess_candidates() before this one.
    And then follow up with crosscheck_tess_candidates()

    Parameters
    ----------
    new_targets: List of strings
        If not None, only fetch aliases for the given targets.

    Examples
    --------
    >>> from gen_tso.catalogs import fetch_catalogs as fetch_cat
    >>> new_targets = fetch_cat.fetch_nasa_tess_candidates()
    >>> aliases = fetch_cat.fetch_tess_aliases(new_targets)
    >>> fetch_cat.crosscheck_tess_candidates()
    """
    candidates = load_targets('tess_candidates_tmp.txt')
    if new_targets is None:
        new_targets = np.unique([target.planet for target in candidates])

    if os.path.exists(f'{ROOT}data/tess_data.txt'):
        known_candidates = load_targets('tess_data.txt')
        known_tess = [target.planet for target in known_candidates]
    else:
        known_tess = []

    # New TESS hosts that are not in confirmed list
    hosts = np.unique([
        target.host
        for target in candidates
        if target.planet in new_targets or target.planet not in known_tess
    ])


    # Get previously known aliases
    known_aliases = {}
    aliases_file = f'{ROOT}data/target_aliases.txt'
    if os.path.exists(aliases_file):
        known_aliases = load_aliases('system')
    output_file = f'{ROOT}data/tess_aliases.pickle'
    # Get previously known aliases
    if os.path.exists(output_file):
        with open(output_file, 'rb') as handle:
            prev_aliases = pickle.load(handle)
        for host,system in prev_aliases.items():
            known_aliases[host] = system

    known_hosts = np.unique([target.host for target in candidates])
    for host in list(known_aliases):
        if host not in known_hosts:
            known_aliases.pop(host)

    # Get new aliases
    aliases = fetch_aliases(hosts, output_file, known_aliases)
    return aliases


def crosscheck_tess_candidates(ncpu=None):
    """
    Do a final TESS-confirmed cross-check and hunt for their Ks mag.
    Write output to tess_data.txt file.

    Before calling this function, you want to run
    fetch_nasa_tess_candidates() and crosscheck_tess_candidates()

    Examples
    --------
    >>> from gen_tso.catalogs import fetch_catalogs as fetch_cat
    >>> new_targets = fetch_cat.fetch_nasa_tess_candidates()
    >>> fetch_cat.fetch_tess_aliases(new_targets)
    >>> fetch_cat.crosscheck_tess_candidates()
    """
    if ncpu is None:
        ncpu = mp.cpu_count()

    planet_aliases = {}
    host_aliases = {}
    # previously known aliases
    aliases_file = f'{ROOT}data/target_aliases.txt'
    if os.path.exists(aliases_file):
        known_aliases = load_aliases('system')
        for host, system in known_aliases.items():
            planet_aliases.update(system['planet_aliases'])
            for alias in system['host_aliases']:
                host_aliases[alias] = host
    # known confirmed aliases
    with open(f'{ROOT}data/nea_aliases.pickle', 'rb') as handle:
        known_aliases = pickle.load(handle)
    for host, system in known_aliases.items():
        planet_aliases.update(system['planet_aliases'])
        for alias in system['host_aliases']:
            host_aliases[alias] = host
    # known TESS aliases
    with open(f'{ROOT}data/tess_aliases.pickle', 'rb') as handle:
        tess_aliases = pickle.load(handle)
    for host, system in tess_aliases.items():
        planet_aliases.update(system['planet_aliases'])
        for alias in system['host_aliases']:
            host_aliases[alias] = host

    # Identity alias for targets without aliases
    candidates = load_targets('tess_candidates_tmp.txt')
    for target in candidates:
        if target.host not in host_aliases:
            host_aliases[target.host] = target.host
        if target.planet not in planet_aliases:
            planet_aliases[target.planet] = target.planet
    aka = u.invert_aliases(host_aliases)

    # Cross-check with confirmed targets:
    confirmed_planets = [target.planet for target in load_targets()]
    candidates = [
        target for target in candidates
        if planet_aliases[target.planet] not in confirmed_planets
    ]
    # Make sure names are NEA names:
    for i,target in enumerate(candidates):
        target.host = host_aliases[target.host]
        target.planet = planet_aliases[target.planet]


    # Now I need to collect the Ks-band magnitudes:
    # Zeroth idea, check if I already have the Ks mag:
    if os.path.exists(f'{ROOT}data/tess_data.txt'):
        known_candidates = load_targets('tess_data.txt')
        known_hosts = [target.host for target in known_candidates]
        for target in candidates:
            if target.host in known_hosts:
                idx = known_hosts.index(target.host)
                target.ks_mag = known_candidates[idx].ks_mag

    # First idea, search in simbad using best known alias to get Ks magnitude
    catalogs = ['2MASS', 'Gaia DR3', 'Gaia DR2', 'TOI']
    names = [
        u.select_alias(aka[target.host], catalogs)
        for i,target in enumerate(candidates)
    ]
    aliases, kmags = fetch_simbad_aliases(names)
    for i,target in enumerate(candidates):
        if not np.isfinite(target.ks_mag) and np.isfinite(kmags[i]):
            target.ks_mag = kmags[i]

    # Plan B, batch search in vizier/2MASS catalog:
    hosts = np.array([
        u.select_alias(aka[target.host], catalogs, target.host)
        for target in candidates
    ])
    missing_targets = [
        target for i,target in enumerate(candidates)
        if np.isnan(target.ks_mag)
        if hosts[i].startswith('2MASS')
    ]
    ra = np.array([target.ra for target in missing_targets]) * deg
    dec = np.array([target.dec for target in missing_targets]) * deg

    catalog = 'II/246/out'
    vizier = Vizier(
        catalog=catalog,
        columns=['RAJ2000', 'DEJ2000', '2MASS', 'Kmag'],
    )
    two_mass_targets = Table(
        [ra, dec],
        names=('_RAJ2000', '_DEJ2000'),
    )
    results = vizier.query_region(two_mass_targets, radius=5.0*arcsec)
    data = results[catalog].as_array().data
    vizier_names = [f'2MASS J{d[3]}' for d in data]
    vizier_ks_mag = [d[4] for d in data]

    for target in candidates:
        host = u.select_alias(aka[target.host], catalogs)
        if host in vizier_names:
            idx = vizier_names.index(host)
            target.ks_mag = vizier_ks_mag[idx]

    # Last resort, scrap from the NEA website
    missing_hosts = np.unique([
        target.host for target in candidates
        if np.isnan(target.ks_mag)
    ])
    with mp.get_context('fork').Pool(ncpu) as pool:
        scrap_ks = pool.map(scrap_nea_kmag, missing_hosts)
    for target in candidates:
        if target.host in missing_hosts:
            idx = list(missing_hosts).index(target.host)
            target.ks_mag = scrap_ks[idx]

    # Save as plain text:
    catalog_file = f'{ROOT}data/tess_data.txt'
    save_catalog(candidates, catalog_file)


def scrap_nea_kmag(target):
    """
    >>> target = 'TOI-5290'
    >>> target = 'TOI-4345'
    >>> scrap_nea_kmag(target)
    """
    response = requests.get(
        url=f'https://exoplanetarchive.ipac.caltech.edu/overview/{target}',
    )
    kmag = np.nan
    if not response.ok:
        print(f'ERROR {target}')
        return kmag

    soup = BeautifulSoup(response.content, 'html.parser')
    pm = '±'
    for dd in soup.find_all('dd'):
        texts = dd.get_text().split()
        if 'mKs' in texts:
            #print(target, dd.text.split())
            kmag_text = texts[-1]
            if kmag_text == '---':
                kmag = 0.0
            elif pm in kmag_text:
                kmag = float(kmag_text[0:kmag_text.find(pm)])
            else:
                kmag = float(kmag_text)
    return kmag


def fetch_gaia_targets(
        ra_source, dec_source, max_separation=80.0, raise_errors=True,
    ):
    """
    Search for Gaia DR3 stellar sources around a given target.

    Parameters
    ----------
    ra_source: Float
        Right ascension (deg) of target to query around.
    dec_source: Float
        Declination (deg) of target to query around.
    max_angular_distance: Float
        Maximum angular distance from target to consider (arcsec).
        Consider that the visit splitting distance for NIRSpec TA is 38"
    raise_errors: Bool
        If True and there was an error while requesting the data,
        raise the error.
        If False and there was an error, print error to screen and
        return a string identifying some known error types.

    Returns
    -------
    names: 1D string array
        Gaia DR3 stellar source names within max_separation from target
    G_mag: 1D float array
        Gaia G magnitude of found stellar sources
    teff: 1D float array
        Effective temperature (K) of found stellar source
    log_g: 1D float array
        log(g) of found stellar sources
    ra: 1D float array
        Right ascension (deg) of found stellar sources
    dec: 1D float array
        Declination (deg) of found stellar sources
    separation: 1D float array
        Angular separation (arcsec) of found stellar sources from target

    Examples
    --------
    >>> import gen_tso.catalogs as cat

    >>> # Stellar sources around WASP-69:
    >>> ra_source = 315.0259661
    >>> dec_source = -5.094857
    >>> cat.fetch_gaia_targets(ra_source, dec_source)
    """
    # Moved inside function to avoid hanging at import time
    # (when astroquery is not reachable)
    from astroquery.gaia import Gaia
    max_sep_degrees = max_separation / 3600.0

    try:
        job = Gaia.launch_job_async(
            f"""
            SELECT * \
            FROM gaiadr3.gaia_source \
            WHERE CONTAINS(\
                POINT(gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec),
                CIRCLE({ra_source}, {dec_source}, {max_sep_degrees}))=1;""",
            dump_to_file=False,
        )
    except Exception as e:
        err_text = f"Gaia astroquery request failed with {e.__class__.__name__}"
        if isinstance(e, socket.gaierror):
            print(
                f"\n{err_text}\n{str(e)}\n"
                "Likely there's no internet connection at the moment\n"
            )
            exception = 'gaierror'
        elif isinstance(e, requests.exceptions.HTTPError):
            print(
                f"\n{err_text}\n{str(e)}\n"
                f"Probably the ESA server is down at the moment\n"
            )
            exception = 'gaierror'
        elif isinstance(e, ssl.SSLError):
            print(
                f"\n{err_text}\n{str(e)}\n"
                "If you got a 'SSL: CERTIFICATE_VERIFY_FAILED' error on an "
                "OSX machine, try following the steps on this link: "
                "https://stackoverflow.com/a/42334357 which will point you to "
                "the ReadMe.rtf file in your Applications/Python 3.X folder\n"
            )
            exception = 'ssl'
        else:
            print(f"\n{err_text}\n{str(e)}")
            exception = 'other'
        if raise_errors:
            raise e
        return exception

    resp = job.get_results()
    targets = resp[~resp['teff_gspphot'].mask]

    c1 = SkyCoord(ra_source, dec_source, unit='deg', frame='icrs')
    separation = []
    for i,target in enumerate(targets):
        c2 = SkyCoord(target['ra'], target['dec'], unit='deg', frame='icrs')
        sep = c1.separation(c2).to('arcsec').value
        separation.append(sep)

    sep_isort = np.argsort(separation)
    for i,idx in enumerate(sep_isort):
        target = targets[idx]

    return (
        targets['designation'].data.data[sep_isort],
        targets['phot_g_mean_mag'].data.data[sep_isort],
        targets['teff_gspphot'].data.data[sep_isort],
        targets['logg_gspphot'].data.data[sep_isort],
        targets['ra'].data.data[sep_isort],
        targets['dec'].data.data[sep_isort],
        np.array(separation)[sep_isort],
    )

