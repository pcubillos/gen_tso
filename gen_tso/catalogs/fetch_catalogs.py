# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'fetch_trexolist',
    'fetch_nea_confirmed_targets',
    'fetch_nea_tess_candidates',
    'fetch_nea_aliases',
    'fetch_simbad_aliases',
    'fetch_vizier_ks',
    'fetch_aliases',
    'fetch_tess_aliases',
]

# TBD: Figure a way to circumvent the grequests conflict  with shiny
#import grequests
import requests
import multiprocessing as mp
from datetime import datetime, timezone
import urllib
import pickle
import warnings
import re

import numpy as np
from astroquery.simbad import Simbad as simbad
from astroquery.vizier import Vizier
from astropy.table import Table
from astropy.units import arcsec, deg
from bs4 import BeautifulSoup
import pyratbay.constants as pc
import pyratbay.atmosphere as pa

# While developing
if False:
    from gen_tso.utils import ROOT
    import gen_tso.catalogs.catalog_utils as u
    from gen_tso.catalogs.source_catalog import load_targets_table

from ..utils import ROOT
from .source_catalog import load_targets_table
from . import catalog_utils as u


def update_databases():
    # Update trexolist database
    fetch_trexolist()

    # Update NEA confirmed targets and their aliases
    fetch_nea_confirmed_targets()
    # Fetch confirmed aliases
    nea_data = load_targets_table()
    hosts = np.unique(nea_data[1])
    output_file = f'{ROOT}data/nea_aliases.pickle'
    fetch_aliases(hosts, output_file)

    # Update NEA TESS candidates and their aliases
    fetch_nea_tess_candidates()
    fetch_tess_aliases()

    # Update aliases list
    curate_aliases()


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
        if np.any(np.in1d(aliases, host_aliases)):
            children.append(planet)

    aliases = {
        alias:planet
        for alias,planet in planet_aliases.items()
        if planet in children
    }
    return aliases


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

    jwst_names = list(u.get_trexolists_targets())
    # Ensure to match against NEA host names for jwst targets
    for host,system in aliases.items():
        is_in = np.in1d(system['host_aliases'], jwst_names)
        if np.any(is_in) and system['host'] not in jwst_names:
            jwst_names.append(system['host'])

    prefixes = jwst_names
    prefixes += ['WASP', 'KELT', 'HAT', 'MASCARA', 'TOI', 'XO', 'TrES']
    kept_aliases = {}
    for host,system in aliases.items():
        for alias,planet in system['planet_aliases'].items():
            for prefix in prefixes:
                if alias.startswith(prefix) and alias != planet:
                    kept_aliases[alias] = planet


    aka = u.invert_aliases(kept_aliases)
    to_remove = []
    for name, aliases in aka.items():
        # Remove candidate if lettered-name exist:
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

    with open(f'{ROOT}data/nea_aliases.txt', 'w') as f:
        for name,aliases in aka.items():
            str_aliases = ','.join(aliases)
            f.write(f'{name}:{str_aliases}\n')


def fetch_trexolist():
    url = 'https://www.stsci.edu/~nnikolov/TrExoLiSTS/JWST/trexolists.csv'
    query_parameters = {}
    response = requests.get(url, params=query_parameters)

    if not response.ok:
        raise ValueError('Could not download TrExoLiSTS database')

    trexolists_path = f'{ROOT}data/trexolists.csv'
    with open(trexolists_path, mode="wb") as file:
        file.write(response.content)

    today = datetime.now(timezone.utc)
    with open(f'{ROOT}data/last_updated_trexolist.txt', 'w') as f:
        f.write(f'{today.year}_{today.month:02}_{today.day:02}')


def fetch_nea_confirmed_targets():
    """
    Fetch (web request) the entire NASA Exoplanet Archive database
    for confirmed planets (there is another one for TESS candidates)
    """
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
        planets[i] = u.complete_entry(resp[j].copy())
        dups = [resp[k] for k in idx_duplicates if k!=j]
        rank = u.rank_planets(dups)
        # Fill the gaps if any
        for j in rank:
            entry = u.complete_entry(dups[j])
            for field in planet.keys():
                if planets[i][field] is None and entry[field] is not None:
                    planets[i][field] = entry[field]
        planets[i] = u.complete_entry(planets[i])

    # Save as plain text:
    with open(f'{ROOT}data/nea_data.txt', 'w') as f:
        host = ''
        for entry in planets:
            ra = entry['ra']
            dec = entry['dec']
            ks_mag = entry['sy_kmag']
            planet = entry['pl_name']
            tr_dur = entry['pl_trandur']
            teff = u.as_str(entry['st_teff'], '.1f')
            logg = u.as_str(entry['st_logg'], '.3f')
            rprs = u.as_str(entry['pl_ratror'], '.3f')
            missing_info = (
                entry['st_teff'] is None or
                entry['st_rad'] is None or
                entry['pl_orbsmax'] is None
            )
            if missing_info:
                teq = 'None'
            else:
                teq, _ = pa.equilibrium_temp(
                    entry['st_teff'],
                    entry['st_rad']*pc.rsun,
                    entry['pl_orbsmax']*pc.au,
                )
                teq = f'{teq:.1f}'

            if entry['hostname'] != host:
                host = entry['hostname']
                f.write(f">{host}: {ra} {dec} {ks_mag} {teff} {logg}\n")
            f.write(f" {planet}: {tr_dur} {rprs} {teq}\n")

    today = datetime.now(timezone.utc)
    with open(f'{ROOT}data/last_updated_nea.txt', 'w') as f:
        f.write(f'{today.year}_{today.month:02}_{today.day:02}')



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

    >>> host_aliases, planet_aliases = cat.fetch_nea_aliases('WASP-999')
    """
    # TBD: Need to import grequest (before requests) to work properly,
    # but that breaks shiny
    if isinstance(targets, str):
        targets = [targets]
    ntargets = len(targets)

    urls = np.array([
        'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/Lookup/'
        f'nph-aliaslookup.py?objname={urllib.parse.quote(target)}'
        for target in targets
    ])

    fetch_status = np.tile(2, ntargets)
    responses = np.tile({}, ntargets)
    batch_size = 25
    n_attempts = 0
    while np.any(fetch_status>0) and n_attempts < 10:
        n_attempts += 1
        mask = fetch_status > 0
        rs = (grequests.get(u) for u in urls[mask])
        resps = iter(grequests.map(rs, size=batch_size))

        for i in range(ntargets):
            if fetch_status[i] <= 0:
                continue
            r = next(resps)
            if r is None:
                continue
            if not r.ok:
                warnings.warn(f"Alias fetching failed for '{targets[i]}'")
                fetch_status[i] -= 1
                continue
            responses[i] = r.json()
            fetch_status[i] = 0
        fetched = np.sum(fetch_status <= 0)
        print(f'Fetched {fetched}/{ntargets} entries on {n_attempts} try')

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


def fetch_simbad_aliases(target, verbose=True):
    """
    Fetch target aliases and Ks magnitude as known by Simbad.

    Examples
    --------
    >>> from gen_tso.catalogs.update_catalogs import fetch_simbad_aliases
    >>> aliases, ks_mag = fetch_simbad_aliases('WASP-69b')
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
            print(f'no Simbad entry for target {repr(target)}')
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
    if not np.isfinite(kmag):
        kmag = None
    return host_alias, kmag


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
        return None

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
        return None
    return None


def fetch_aliases(hosts, output_file=None):
    """
    Fetch known aliases from the NEA and Simbad databases for a list
    of host stars.  Store output dictionary of aliases to pickle file.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> from gen_tso.utils import ROOT

    >>> # Confirmed targets
    >>> nea_data = cat.load_targets_table()
    >>> hosts = np.unique(nea_data[1])
    >>> output_file = f'{ROOT}data/nea_aliases.pickle'
    >>> cat.fetch_aliases(hosts, output_file)
    """
    host_aliases, planet_aliases = fetch_nea_aliases(hosts)

    # Keep track of trexolists aliases to cross-check:
    jwst_names = u.get_trexolists_targets()
    jwst_aliases = u.get_trexolists_targets(grouped=True)

    aliases = {}
    nhosts = len(hosts)
    for i in range(nhosts):
        # Isolate host-planet(s) aliases
        stars = np.unique(list(host_aliases[i].values()))
        hosts_aka = u.invert_aliases(host_aliases[i])
        for host, h_aliases in hosts_aka.items():
            if hosts[i] in h_aliases:
                host_name = host
                break

        h_aliases = [
            alias
            for alias,host in host_aliases[i].items()
            if host == host_name
        ]
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

        in_jwst = np.in1d(h_aliases, jwst_names)
        if np.any(in_jwst):
            j_alias = np.array(h_aliases)[in_jwst]

        # Complement with Simbad aliases:
        s_aliases, kmag = fetch_simbad_aliases(host_name, verbose=False)
        new_aliases = []
        for alias in s_aliases:
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
        in_jwst = np.in1d(h_aliases, jwst_names)
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
            #new_planets = ''
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
                    #new_planets += f'\n--> {planet_name}  ({planet})'
                    p_aliases[planet_name] = planet
            #if new_planets != '':
            #    print(f'\n[{i}]  {hosts[i]}{new_planets}')
            #    for kid in pals:
            #        print(f'    {kid}')

        system = {
            'host': host_name,
            'planets': children_names,
            'host_aliases': h_aliases,
            'planet_aliases': p_aliases,
        }
        aliases[host_name] = system

    if output_file is not None:
        with open(output_file, 'wb') as handle:
            pickle.dump(aliases, handle, protocol=4)
    return aliases


def fetch_nea_tess_candidates():
    """
    Fetch entries in the NEA TESS candidates table.
    Remove already confirmed targets.
    """
    r = requests.get(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
        "select+toi,toipfx,pl_trandurh,pl_trandep,pl_rade,pl_eqt,ra,dec,"
        "st_tmag,st_teff,st_logg,st_rad,tfopwg_disp+"
        "from+toi+"
        "&format=json"
    )
    if not r.ok:
        raise ValueError("Something's not OK")

    resp = r.json()
    tess_hosts = [f"TOI-{entry['toipfx']}" for entry in resp]
    tess_planets = [f"TOI-{entry['toi']}" for entry in resp]
    status = [entry['tfopwg_disp'] for entry in resp]
    ntess = len(tess_planets)

    # Discard confirmed planets:
    nea_data = load_targets_table()
    targets = nea_data[0]
    confirmed_hosts = nea_data[1]
    ks_mag = nea_data[4]

    # Unpack known aliases:
    with open(f'{ROOT}data/nea_aliases.pickle', 'rb') as handle:
        nea_aliases = pickle.load(handle)
    planet_aliases = {}
    host_aliases = {}
    for host, system in nea_aliases.items():
        planet_aliases.update(system['planet_aliases'])
        for alias in system['host_aliases']:
            host_aliases[alias] = host

    j, k, l = 0, 0, 0
    is_candidate = np.ones(ntess, bool)
    tess_mag = np.zeros(ntess)
    for i, planet in enumerate(tess_planets):
        # Update names if possible:
        tess_host = tess_hosts[i]
        if tess_host in host_aliases:
            tess_host = tess_hosts[i] = host_aliases[tess_host]
        if planet in planet_aliases:
            planet = tess_planets[i] = planet_aliases[planet]

        if planet in targets:
            is_candidate[i] = False
            j += 1
            # print(f"[{j}] '{planet}' is a confirmed target.")
            # 961 planets
            continue
        if tess_host in confirmed_hosts:
            k += 1
            target_idx = confirmed_hosts.index(tess_host)
            tess_mag[i] = ks_mag[target_idx]
            # print(f"[{k}] '{planet}' orbits known star: '{tess_host}' (ks={tess_mag[i]})")
            # 19 hosts
        if status[i] in ['FA', 'FP']:
            is_candidate[i] = False
            l += 1


    # Save raw data:
    tess_planets = np.array(tess_planets)
    tess_hosts = np.array(tess_hosts)
    ra = np.array([entry['ra'] for entry in resp])
    dec = np.array([entry['dec'] for entry in resp])
    teff = np.array([entry['st_teff'] for entry in resp])
    logg = np.array([entry['st_logg'] for entry in resp])
    tr_dur = np.array([entry['pl_trandurh'] for entry in resp])
    rprs = np.array([np.sqrt(entry['pl_trandep']) for entry in resp])
    teq = np.array([entry['pl_eqt'] for entry in resp])

    candidates = dict(
        planets=tess_planets[is_candidate],
        hosts=tess_hosts[is_candidate],
        ra=ra[is_candidate],
        dec=dec[is_candidate],
        teff=teff[is_candidate],
        logg=logg[is_candidate],
        tr_dur=tr_dur[is_candidate],
        rprs=rprs[is_candidate],
        teq=teq[is_candidate],
        ks_mag=tess_mag[is_candidate],
    )
    with open(f'{ROOT}data/nea_tess_candidates_raw.pickle', 'wb') as handle:
        pickle.dump(candidates, handle, protocol=4)



def fetch_tess_aliases(ncpu=None):
    """
    Get TESS aliases and also finalize the tess database.
    """
    if ncpu is None:
        ncpu = mp.cpu_count()

    # Known aliases:
    with open(f'{ROOT}data/nea_aliases.pickle', 'rb') as handle:
        nea_aliases = pickle.load(handle)
    # Candidates
    with open(f'{ROOT}data/nea_tess_candidates_raw.pickle', 'rb') as handle:
        candidates = pickle.load(handle)
    tess_planets = candidates['planets']
    tess_hosts = candidates['hosts']
    ks_mag = candidates['ks_mag']
    ntess = len(tess_planets)

    # Get the tess aliases
    hosts = np.unique([
        host for host in tess_hosts
        if host not in nea_aliases
    ])
    tess_aliases_file = f'{ROOT}data/tess_aliases.pickle'
    tess_aliases = fetch_aliases(hosts, tess_aliases_file)


    # Unpack known aliases:
    planet_aliases = {}
    host_aliases = {}
    for host, system in nea_aliases.items():
        planet_aliases.update(system['planet_aliases'])
        for alias in system['host_aliases']:
            host_aliases[alias] = host
    for host, system in tess_aliases.items():
        planet_aliases.update(system['planet_aliases'])
        for alias in system['host_aliases']:
            host_aliases[alias] = host
    aka = u.invert_aliases(host_aliases)

    # First idea, search in simbad using best known alias to get Ks magnitude
    catalogs = ['2MASS', 'Gaia DR3', 'Gaia DR2', 'TOI']
    k = 0
    for i,planet in enumerate(tess_planets):
        tess_host = tess_hosts[i]
        if tess_host in host_aliases and tess_host != host_aliases[tess_host]:
            tess_host = tess_hosts[i] = host_aliases[tess_host]
        if planet in planet_aliases and planet != planet_aliases[planet]:
            planet = tess_planets[i] = planet_aliases[planet]

        if ks_mag[i] > 0:
            continue

        name = u.select_alias(aka[tess_host], catalogs)
        if i%500 == 0:
            print(f"~~ [{i}] Searching for '{tess_host}' / '{name}' ~~")
        aliases, kmag = fetch_simbad_aliases(name, verbose=False)
        if kmag is not None:
            k += 1
            ks_mag[i] = kmag

    # Plan B, batch search in vizier catalog:
    two_mass_hosts = np.array([
        u.select_alias(aka[host], catalogs, host)
        for host in tess_hosts
    ])
    mask = [
        host.startswith('2M') and ks_mag[i]==0
        for i,host in enumerate(two_mass_hosts)
    ]
    two_mass_hosts = two_mass_hosts[mask]
    ra = candidates['ra'][mask]
    dec = candidates['dec'][mask]

    catalog = 'II/246/out'
    vizier = Vizier(
        catalog=catalog,
        columns=['RAJ2000', 'DEJ2000', '2MASS', 'Kmag'],
    )
    two_mass_targets = Table(
        [ra*deg, dec*deg],
        names=('_RAJ2000', '_DEJ2000'),
    )
    results = vizier.query_region(two_mass_targets, radius=5.0*arcsec)

    data = results[catalog].as_array().data
    vizier_names = [d[3] for d in data]
    for i, tess_host in enumerate(tess_hosts):
        host_alias = u.select_alias(aka[tess_host], catalogs)
        if host_alias in two_mass_hosts:
            idx = vizier_names.index(host_alias[-16:])
            ks_mag[i] = data[idx][4]


    # Plan C, search in vizier catalog one by one:
    missing_hosts = [host for host,ks in zip(tess_hosts,ks_mag) if ks==0.0]
    missing_hosts = np.unique(missing_hosts)
    missing_hosts = [
        u.select_alias(aka[host], catalogs)
        for host in missing_hosts
    ]
    with mp.get_context('fork').Pool(ncpu) as pool:
        vizier_ks = pool.map(fetch_vizier_ks, missing_hosts)

    for i, tess_host in enumerate(tess_hosts):
        alias_host = u.select_alias(aka[tess_host], catalogs)
        if alias_host in missing_hosts:
            idx = list(missing_hosts).index(alias_host)
            if vizier_ks[idx] is not None:
                ks_mag[i] = vizier_ks[idx]

    # Last resort, scrap from the NEA pages
    missing_hosts = [host for host,ks in zip(tess_hosts,ks_mag) if ks==0.0]
    missing_hosts = np.unique(missing_hosts)

    with mp.get_context('fork').Pool(ncpu) as pool:
        scrap_ks = pool.map(scrap_nea_kmag, missing_hosts)
    for i,planet in enumerate(tess_planets):
        tess_host = tess_hosts[i]
        if tess_host in missing_hosts:
            idx = list(missing_hosts).index(tess_host)
            if scrap_ks[idx] is not None:
                ks_mag[i] = scrap_ks[idx]


    # Save as plain text:
    with open(f'{ROOT}data/tess_data.txt', 'w') as f:
        host = ''
        for i in range(ntess):
            ra = candidates['ra'][i]
            dec = candidates['dec'][i]
            ksmag = f'{ks_mag[i]:.3f}' if ks_mag[i]>0.0 else 'None'
            planet = tess_planets[i]
            tr_dur = candidates['tr_dur'][i]
            teff = u.as_str(candidates['teff'][i], '.1f')
            logg = u.as_str(candidates['logg'][i], '.3f')
            depth = candidates['rprs'][i] **2.0 * pc.ppm
            rprs = u.as_str(np.sqrt(depth), '.3f')
            teq = u.as_str(candidates['teq'][i], '.1f')

            if tess_hosts[i] != host:
                host = tess_hosts[i]
                f.write(f">{host}: {ra} {dec} {ksmag} {teff} {logg}\n")
            f.write(f" {planet}: {tr_dur} {rprs} {teq}\n")

    today = datetime.now(timezone.utc)
    with open(f'{ROOT}data/last_updated_tess.txt', 'w') as f:
        f.write(f'{today.year}_{today.month:02}_{today.day:02}')


def scrap_nea_kmag(target):
    """
    >>> target = 'TOI-5290'
    >>> target = 'TOI-4345'
    >>> scrap_nea_kmag(target)
    """
    response = requests.get(
        url=f'https://exoplanetarchive.ipac.caltech.edu/overview/{target}',
    )
    if not response.ok:
        print(f'ERROR {target}')
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    pm = '±'

    kmag = None
    for dd in soup.find_all('dd'):
        texts = dd.get_text().split()
        if 'mKs' in texts:
            print(target, dd.text.split())
            kmag_text = texts[-1]
            if kmag_text == '---':
                kmag = 0.0
            elif pm in kmag_text:
                kmag = float(kmag_text[0:kmag_text.find(pm)])
            else:
                kmag = float(kmag_text)
    return kmag

