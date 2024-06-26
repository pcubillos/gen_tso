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

import sys
# Skip importing grequests when launching the app because it breaks shiny
if 'bin/tso' not in sys.argv[0]:
    import grequests
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

from ..utils import ROOT
from .catalogs import load_targets, load_trexolists
from . import utils as u
from . import target as tar
from .target import Target


def save_catalog(targets, catalog_file):
    """
    save_catalog(targets, catalog_file)
    """
    # Save as plain text:
    with open(catalog_file, 'w') as f:
        f.write(
            '# > host: RA(deg) dec(deg) Ks_mag '
            'rstar(rsun) mstar(msun) teff(K) log_g metallicity(dex)\n'
            '# planet: T14(h) rplanet(rearth) mplanet(mearth) '
            'semi-major_axis(AU) period(d) t_eq(K) is_min_mass\n'
        )
        host = ''
        for target in targets:
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


def update_databases():
    """
    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> cat.fetch_trexolist()
    """
    # Update trexolist database
    fetch_trexolist()

    # Update NEA confirmed targets and their aliases
    fetch_nea_confirmed_targets()
    # Fetch confirmed aliases
    targets = load_targets()
    hosts = np.unique([target.host for target in targets])
    output_file = f'{ROOT}data/nea_aliases.pickle'
    fetch_aliases(hosts, output_file)

    # Update NEA TESS candidates and their aliases
    fetch_nea_tess_candidates()
    fetch_tess_aliases()

    # Update aliases list
    curate_aliases()


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

    jwst_names = list(load_trexolists())
    # Ensure to match against NEA host names for jwst targets
    for host,system in aliases.items():
        is_in = np.in1d(system['host_aliases'], jwst_names)
        if np.any(is_in) and system['host'] not in jwst_names:
            jwst_names.append(system['host'])

    prefixes = jwst_names
    prefixes += ['WASP', 'KELT', 'HAT', 'MASCARA', 'TOI', 'XO', 'TrES']
    keep_aliases = {}
    for host,system in aliases.items():
        for alias,planet in system['planet_aliases'].items():
            for prefix in prefixes:
                if alias.startswith(prefix) and alias != planet:
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

    with open(f'{ROOT}data/target_aliases.txt', 'w') as f:
        for name,aliases in aka.items():
            str_aliases = ','.join(aliases)
            f.write(f'{name}:{str_aliases}\n')


def fetch_trexolist():
    """
    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> cat.fetch_trexolist()
    """
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
    fillers = []
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
        # Solve stellar parameters
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
            t = tar.rank_planets(target, entries)
            fillers.append(t)
            targets.append(target)

    catalog_file = f'{ROOT}data/nea_data.txt'
    save_catalog(targets, catalog_file)

    today = datetime.now(timezone.utc)
    with open(f'{ROOT}data/last_updated_nea.txt', 'w') as f:
        f.write(f'{today.year}_{today.month:02}_{today.day:02}')

    # Some stats
    # np.unique(fillers, return_counts=True)
    # (array([0, 1, 2]), array([4418, 1202,   18]))
    # (array([0, 1, 2]), array([3145, 2466,   27]))
    # (array([0, 1, 2]), array([4528, 1093,   17]))


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
    #simbad.add_votable_fields("fe_h")

    host_alias = []
    kmag = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simbad_info = simbad.query_object(target)
    if simbad_info is None:
        if verbose:
            print(f'no Simbad entry for target {repr(target)}')
        return host_alias, kmag

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
            return host_alias, kmag
        # go after star
        simbad_info = simbad.query_object(host)
        if simbad_info is None:
            if verbose:
                print(f'Simbad host {repr(host)} not found')
            return host_alias, kmag

    host_info = simbad_info['IDS'].value.data[0]
    host_alias = host_info.split('|')
    kmag = simbad_info['FLUX_K'].value.data[0]
    # fetch metallicity?
    if not np.isfinite(kmag):
        kmag = np.nan
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


def fetch_aliases(hosts, output_file=None):
    """
    Fetch known aliases from the NEA and Simbad databases for a list
    of host stars.  Store output dictionary of aliases to pickle file.

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
    host_aliases, planet_aliases = fetch_nea_aliases(hosts)

    # Keep track of trexolists aliases to cross-check:
    jwst_names = load_trexolists()
    jwst_aliases = load_trexolists(grouped=True)

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
        "st_tmag,st_teff,st_logg,st_rad,pl_orbper,tfopwg_disp+"
        "from+toi+"
        "&format=json"
    )
    if not r.ok:
        raise ValueError("Something's not OK")

    entries = [format_nea_entry(entry) for entry in r.json()]
    ntess = len(entries)
    status = [entry['tfopwg_disp'] for entry in entries]

    # Discard confirmed planets:
    database = 'nea_data.txt'
    targets = load_targets(database)
    confirmed_targets = [target.planet for target in targets]
    confirmed_hosts = [target.host for target in targets]

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
    tess_targets = []
    for i in range(ntess):
        target = Target(entries[i])
        # Update names if possible:
        if target.host in host_aliases:
            target.host = host_aliases[target.host]
        if target.planet in planet_aliases:
            target.planet = planet_aliases[target.planet]

        if target.planet in confirmed_targets:
            is_candidate[i] = False
            j += 1
            continue
        if target.host in confirmed_hosts:
            # Update with star props
            k += 1
            idx = confirmed_hosts.index(target.host)
            star = targets[idx]
            target.copy_star(star)
        if status[i] in ['FA', 'FP']:
            is_candidate[i] = False
            l += 1
            continue
        tess_targets.append(target)

    # Save temporary data (still need to hunt for Ks mags):
    catalog_file = f'{ROOT}data/tess_candidates_tmp.txt'
    save_catalog(tess_targets, catalog_file)


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
    candidates = load_targets('tess_candidates_tmp.txt')
    ntess = len(candidates)

    # Get the tess aliases
    hosts = np.unique([
        target.host for target in candidates
        if target.host not in nea_aliases
    ])
    tess_aliases_file = f'{ROOT}data/tess_aliases.pickle'
    tess_aliases = fetch_aliases(hosts, tess_aliases_file)


    # Now I also have tess aliases:
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
    for i in range(ntess):
        target = candidates[i]
        target.host = host_aliases[target.host]
        target.planet = planet_aliases[target.planet]

        if np.isfinite(target.ks_mag):
            continue

        name = u.select_alias(aka[target.host], catalogs)
        if i%500 == 0:
            print(f"~~ [{i}] Searching for '{target.host}' / '{name}' ~~")
        aliases, kmag = fetch_simbad_aliases(name, verbose=False)
        if np.isfinite(kmag):
            k += 1
            target.ks_mag = kmag

    # Plan B, batch search in vizier catalog:
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

    # Plan C, search in vizier catalog one by one:
    missing_hosts = np.unique([
        u.select_alias(aka[target.host], catalogs)
        for target in candidates
        if np.isnan(target.ks_mag)
    ])
    with mp.get_context('fork').Pool(ncpu) as pool:
        vizier_ks = pool.map(fetch_vizier_ks, missing_hosts)

    for target in candidates:
        host = u.select_alias(aka[target.host], catalogs)
        if host in missing_hosts:
            idx = list(missing_hosts).index(host)
            target.ks_mag = vizier_ks[idx]

    # Last resort, scrap from the NEA pages
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
    kmag = np.nan
    if not response.ok:
        print(f'ERROR {target}')
        return kmag

    soup = BeautifulSoup(response.content, 'html.parser')
    pm = '±'
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


