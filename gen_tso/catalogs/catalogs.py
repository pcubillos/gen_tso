# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'find_target',
    'Catalog',
    'load_trexolists',
    'load_targets',
    'load_aliases',
]

from datetime import datetime
from astropy.io import ascii
import numpy as np
import prompt_toolkit as ptk

from ..utils import ROOT
from . import utils as u
from .target import Target


def find_target(targets=None):
    """
    Interactive prompt with tab-completion to search for targets.

    Parameters
    ----------
    targets: list of Target objects

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> target = cat.find_target()
    """
    if targets is None:
        targets = load_targets('nea_data.txt', is_confirmed=True)
    planets = [target.planet for target in targets]
    aliases = []
    for target in targets:
        aliases += target.aliases
    planets += list(aliases)

    completer = ptk.completion.WordCompleter(
        planets,
        sentence=True,
        match_middle=True,
    )
    session = ptk.PromptSession(
        history=ptk.history.FileHistory(f'{ROOT}/data/target_search_history')
    )
    name = session.prompt(
        "(Press 'tab' for autocomplete)\nEnter Planet name: ",
        completer=completer,
        complete_while_typing=False,
    )
    if name in aliases:
        for target in targets:
            if name in target.aliases:
                return target
    if name in planets:
        return targets[planets.index(name)]

    return None


class Catalog():
    """
    Load the entire catalog.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> catalog = cat.Catalog()
    """
    def __init__(self):
        # Confirmed planets and TESS candidates
        nea_targets = load_targets('nea_data.txt', is_confirmed=True)
        tess_targets = load_targets('tess_data.txt', is_confirmed=False)
        self.targets = nea_targets + tess_targets

        # JWST targets
        trexo_data = load_trexolists(grouped=True)
        njwst = len(trexo_data)
        host_aliases = load_aliases('host')

        jwst_hosts = []
        for jwst_target in trexo_data:
            hosts = np.unique([
                host_aliases[host] if host in host_aliases else host
                for host in jwst_target['target']
            ])
            jwst_target['nea_hosts'] = hosts
            jwst_hosts += list(hosts)
        jwst_hosts = np.unique(jwst_hosts)

        planet_aliases = load_aliases('planet')
        planets_aka = u.invert_aliases(planet_aliases)

        for target in self.targets:
            target.is_jwst = target.host in jwst_hosts and target.is_transiting
            if target.is_jwst:
                for j in range(njwst):
                    if target.host in trexo_data[j]['nea_hosts']:
                        break
                target.trexo_data = trexo_data[j]

            if target.planet in planets_aka:
                target.aliases = planets_aka[target.planet]

        self._transit_mask = [target.is_transiting for target in self.targets]
        self._jwst_mask = [target.is_jwst for target in self.targets]
        self._confirmed_mask = [target.is_confirmed for target in self.targets]


    def get_target(
            self, name=None,
            is_transit=True, is_jwst=None, is_confirmed=True,
        ):
        """
        Search by name for a planet in the catalog.

        Parameters
        ----------
        name: String
            If not None, name of the planet to search.
            If None, an interactive prompt will open to search for the planet
        is_transit: Bool
            If True/False restrict search to transiting/non-transiting planets
            If None, consider all targets.
        is_jwst: Bool
            If True/False restrict search to planet of/not JWST hosts
            If None, consider all targets.
        is_confirmed: Bool
            If True/False restrict search to confirmed/candidate planets
            If None, consider all targets.

        Returns
        -------
        target: a Target object
            Target with the system properties of the searched planet.
            If no target was found on the catalog, return None.
        """
        mask = np.ones(len(self.targets), bool)
        if is_transit is not None:
            mask &= np.array(self._transit_mask) == is_transit
        if is_jwst is not None:
            mask &= np.array(self._jwst_mask) == is_jwst
        if is_confirmed is not None:
            mask &= np.array(self._confirmed_mask) == is_confirmed

        targets = [target for target,flag in zip(self.targets,mask) if flag]

        if name is None:
            return find_target(targets)

        target = u.normalize_name(name)
        for target in targets:
            if name == target.planet or name in target.aliases:
                return target

    def show_target(
            self, name=None,
            is_transit=True, is_jwst=None, is_confirmed=True,
        ):
        target = self.get_target(name, is_transit, is_jwst, is_confirmed)
        if target is None:
            return
        print(target)


def load_targets(database='nea_data.txt', is_confirmed=np.nan):
    """
    Unpack star and planet properties from plain text file.

    Parameters
    ----------
    databases: String
        nea_data.txt or tess_data.txt

    Returns
    -------
    targets: List of Target

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> nea_data = cat.load_nea_targets_table()
    """
    # database = 'new_nea_data.txt'
    with open(f'{ROOT}data/{database}', 'r') as f:
        lines = f.readlines()

    lines = [
        line for line in lines
        if not line.strip().startswith('#')
    ]
    targets = []
    for line in lines:
        if line.startswith('>'):
            name_len = line.find(':')
            host = line[1:name_len]
            star_vals = np.array(line[name_len+1:].split(), float)
            ra, dec, ks_mag, rstar, mstar, teff, logg, metal = star_vals
        elif line.startswith(' '):
            name_len = line.find(':')
            planet = line[1:name_len].strip()
            planet_vals = np.array(line[name_len+1:].split(), float)
            t_dur, rplanet, mplanet, sma, period, teq, min_mass = planet_vals

            target = Target(
                host=host,
                mstar=mstar, rstar=rstar, teff=teff, logg_star=logg,
                metal_star=metal,
                ks_mag=ks_mag, ra=ra, dec=dec,
                planet=planet,
                mplanet=mplanet, rplanet=rplanet,
                period=period, sma=sma, transit_dur=t_dur,
                is_confirmed=is_confirmed,
                is_min_mass=bool(min_mass),
            )
            targets.append(target)

    return targets


def load_trexolists(grouped=False, trexo_file=None):
    """
    Get the data from the trexolists.csv file.
    Note that trexolists know targets by their host star, not by
    individual planets in a given system.

    Parameters
    ----------
    grouped: Bool
        - If False, return the a dictionary where each item contains
          a 1D list for each individual program.
        - If True, return list for each individual host, where each entry
          contains a dictionary of 1D lists of all programs for the target.
    trexo_file: String
        If None, extract data from default Gen TSO location.
        Otherwise, a path to a trexolists.csv file.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>>
    >>> # Get data as lists of individual programs:
    >>> trexo = cat.load_trexolists()
    >>> print(trexo['target'])
    ['L 168-9' 'HAT-P-14' 'WASP-80' 'WASP-80' 'WASP-69' 'GJ 436' ...]
    >>>
    >>> print(list(trexo))
    ['target', 'trexo_name', 'program', 'ra', 'dec', 'event', 'mode', 'subarray', 'readout', 'groups', 'phase_start', 'phase_end', 'duration', 'date_start', 'plan_window', 'proprietary_period', 'status']

    >>> # Get data as lists of (host) targets:
    >>> trexo = cat.load_trexolists(grouped=True)
    >>> trexo[19]
    {'target': array(['WASP-43', 'WASP-43'], dtype='<U23'),
    'trexo_name': array(['WASP-43', 'WASP-43'], dtype='<U23'),
    'program': array(['GTO 1224 Birkmann', 'ERS 1366 Batalha'], dtype='<U22'),
    'ra': array(['10:19:37.9634', '10:19:37.9649'], dtype='<U13'),
    'dec': array(['-09:48:23.21', '-09:48:23.19'], dtype='<U12'),
    'event': array(['phase', 'phase'], dtype='<U9'),
    'mode': array(['NIRSPEC BOTS+G395H', 'MIRI LRS'], dtype='<U20'),
    'subarray': array(['SUB2048', 'SLITLESSPRISM'], dtype='<U13'),
    'readout': array(['NRSRAPID', 'FASTR1'], dtype='<U8'),
    'groups': array([20, 64]),
    'phase_start': array([0.18394, 0.13912]),
    'phase_end': array([0.20955, 0.16473]),
    'duration': array([28.57, 31.82]),
    'date_start': array([datetime.datetime(2023, 5, 14, 9, 3, 30),
           datetime.datetime(2022, 11, 30, 23, 36, 1)], dtype=object),
    'plan_window': array(['X', 'X'], dtype='<U21'),
    'proprietary_period': array([12,  0]),
    'status': array(['Archived', 'Archived'], dtype='<U14'),
    'truncated_ra': array('10:19', dtype='<U5'),
    'truncated_dec': array('-09:48', dtype='<U6')}
    """
    if trexo_file is None:
        trexo_file = f'{ROOT}data/trexolists.csv'

    trexolist_data = ascii.read(
        trexo_file,
        format='csv', guess=False, fast_reader=False, comment='#',
    )

    norm_targets = np.array([
        u.normalize_name(target)
        for target in trexolist_data['Target']
    ])
    trexo_data = {
        'target': norm_targets,
        'trexo_name': np.array(trexolist_data['Target'])
    }

    category = trexolist_data['Category']
    programs = trexolist_data['Program']
    pi = trexolist_data['PI name']
    trexo_data['program'] = np.array([
        f"{categ} {prog} {name}"
        for categ,prog,name in zip(category, programs, pi)
    ])

    trexo_data['ra'] = np.array(trexolist_data['R.A. 2000'])
    trexo_data['dec'] = np.array(trexolist_data['Dec. 2000'])

    trexo_data['event'] = np.array([
        event.lower().replace('phasec', 'phase')
        for event in trexolist_data['Event']
    ])
    # see_phase?

    trexo_data['mode'] = np.array([
        obs.replace('.', ' ')
        for obs in trexolist_data['Mode']
    ])
    trexo_data['subarray'] = np.array(trexolist_data['Subarray'])
    trexo_data['readout'] = np.array(trexolist_data['Readout pattern'])
    trexo_data['groups'] = np.array(trexolist_data['Groups'])

    trexo_data['phase_start'] = np.array([
        np.nan if phase=='N/A' else float(phase)
        for phase in trexolist_data['Start.Phase']
    ])
    trexo_data['phase_end'] = np.array([
        np.nan if phase=='N/A' else float(phase)
        for phase in trexolist_data['End.Phase']
    ])
    trexo_data['duration'] = np.array(trexolist_data['Hours'])

    trexo_data['date_start'] = np.array([
        np.nan if date=='X' else datetime.strptime(date,'%b_%d_%Y_%H:%M:%S')
        for date in trexolist_data['Start date']
    ])

    date_start = []
    date_end = []
    plan_window = []
    s_dates = trexolist_data['Start date']
    e_dates = trexolist_data['End date']
    windows = trexolist_data['Plan Windows']
    for i in range(len(norm_targets)):
        try:
            date = datetime.strptime(s_dates[i],'%b_%d_%Y_%H:%M:%S')
        except:
            date = np.nan
        try:
            end = datetime.strptime(e_dates[i],'%b_%d_%Y_%H:%M:%S')
        except:
            end = np.nan
        try:
            window = windows[i][:windows[i].index('-')]
            window = datetime.strptime(window,'%b%d,%Y')
        except:
            window = np.nan
        date_start.append(date)
        date_end.append(end)
        plan_window.append(window)
    trexo_data['date_start'] = np.array(date_start)
    trexo_data['date_end'] = np.array(date_end)
    trexo_data['plan_window'] = np.array(plan_window)

    trexo_data['proprietary_period'] = np.array(trexolist_data['Prop.Period'])
    trexo_data['status'] = np.array(trexolist_data['Status'])

    if not grouped:
        return trexo_data

    # Use RA and dec to detect aliases for a same object
    truncated_ra = np.array([ra[0:5] for ra in trexo_data['ra']])
    truncated_dec = np.array([dec[0:6] for dec in trexo_data['dec']])

    ntargets = len(trexo_data['target'])
    taken = np.zeros(ntargets, bool)
    target_sets = []
    trexo_ra = []
    trexo_dec = []
    for i in range(ntargets):
        if taken[i]:
            continue
        group_indices = [i]
        ra = truncated_ra[i]
        dec = truncated_dec[i]
        trexo_ra.append(ra)
        trexo_dec.append(dec)
        taken[i] = True
        for j in range(i,ntargets):
            if truncated_ra[j]==ra and truncated_dec[j]==dec and not taken[j]:
                group_indices.append(j)
                taken[j] = True
        target_sets.append(group_indices)

    grouped_data = []
    for i,indices in enumerate(target_sets):
        target = {}
        for key in trexo_data.keys():
            target[key] = trexo_data[key][indices]
        target['truncated_ra'] = np.array(trexo_ra[i])
        target['truncated_dec'] = np.array(trexo_dec[i])
        grouped_data.append(target)

    return grouped_data


def parse(name, style):
    """
    Parse a planet name as a planet (no change) or as a star,
    in which case it will identify from confirmed or candidate
    format name.

    Parameters
    ----------
    style: String
        Select from 'planet' or 'host'.

    Returns
    -------
    name: String
        Parsed name.

    Examples
    --------
    >>> from gen_tso.catalogs.catalogs import parse
    >>> parse('WASP-80 b', 'planet')
    WASP-80 b'
    >>> parse('WASP-80 b', 'host')
    WASP-80'
    >>> parse('TOI-316.01', 'host')
    'TOI-316'
    """
    if style == 'planet':
        return name
    elif style == 'host':
        if u.is_letter(name):
            return name[:-2]
        end = name.rindex('.')
        return name[:end]


def load_aliases(style='planet', aliases_file=None):
    """
    Load file with known aliases of NEA targets.

    Parameters
    ----------
    style: String
        Select from 'planet', 'host', or 'system'.

    Returns
    -------
    aliases: Dictionary
        Dictionary of aliases to-from NASA Exoplanet Archive name.
        See below for examples depending on the 'style' argument.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>>
    >>> # From alias planet name to NEA name:
    >>> aliases = cat.load_aliases('planet')
    >>> aliases['CD-38 2551 b']
    'WASP-63 b'
    >>>
    >>> # From alias host name to NEA name:
    >>> aliases = cat.load_aliases('host')
    >>> aliases['CD-38 2551']
    'WASP-63'
    >>>
    >>> # As stellar system with all host and planet aliases:
    >>> aliases = cat.load_aliases('system')
    >>> aliases['WASP-63']
    {'host': 'WASP-63',
     'planets': ['WASP-63 b'],
     'host_aliases': array(['CD-38 2551', 'TOI-483', 'WASP-63'], dtype='<U10'),
     'planet_aliases': {'TOI-483.01': 'WASP-63 b',
      'CD-38 2551 b': 'WASP-63 b',
      'WASP-63 b': 'WASP-63 b'}}
    """
    if style not in ['planet', 'host', 'system']:
        raise ValueError(
            "Invalid alias style, select from: 'planet', 'host', or 'system'"
        )
    if aliases_file is None:
        aliases_file = f'{ROOT}data/target_aliases.txt'

    with open(aliases_file, 'r') as f:
        lines = f.readlines()

    if style != 'system':
        aliases = {}
        for line in lines:
            loc = line.index(':')
            name = parse(line[:loc], style)
            for alias in line[loc+1:].strip().split(','):
                aliases[parse(alias,style)] = name
            aliases[name] = name
        return aliases

    aliases = {}
    current_host = ''
    for line in lines:
        loc = line.index(':')
        planet = parse(line[:loc], 'planet')
        host = parse(line[:loc], 'host')
        host_aliases = [
            parse(name, 'host')
            for name in line[loc+1:].strip().split(',')
        ]
        host_aliases += [host]
        planet_aliases = {
            parse(name, 'planet'): planet
            for name in line[loc+1:].strip().split(',')
        }
        planet_aliases[planet] = planet
        if host != current_host:
            # Save old one
            if current_host != '':
                system['host_aliases'] = np.unique(system['host_aliases'])
                aliases[current_host] = system
            # Start new one
            system = {
                'host': host,
                'planets': [planet],
                'host_aliases': host_aliases,
                'planet_aliases': planet_aliases,
            }
            current_host = host
        else:
            system['host_aliases'] += host_aliases
            system['planets'] += [planet]
            system['planet_aliases'].update(planet_aliases)
    system['host_aliases'] = np.unique(system['host_aliases'])
    aliases[current_host] = system
    return aliases

