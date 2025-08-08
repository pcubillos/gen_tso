# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'find_target',
    'Catalog',
    'load_trexolists',
    'load_programs',
    'load_targets',
    'load_aliases',
    '_group_by_target',
]

import csv
from datetime import datetime
import json

from astropy.io import ascii
import numpy as np
import prompt_toolkit as ptk

from astropy.coordinates import Angle, SkyCoord
from astropy.units import hourangle, deg

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
        targets += load_targets('tess_data.txt', is_confirmed=False)
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
    def __init__(self, custom_targets=None):
        # Confirmed planets and TESS candidates
        nea_targets = load_targets('nea_data.txt', is_confirmed=True)
        tess_targets = load_targets('tess_data.txt', is_confirmed=False)
        self.targets = nea_targets + tess_targets
        if custom_targets is not None:
            custom = load_targets(custom_targets, is_confirmed=True)
            self.targets += custom

        # TBD: a switch between load_trexolists() and load_programs()?
        programs = load_trexolists(grouped=True)
        njwst = len(programs)
        host_aliases = load_aliases('host')

        jwst_hosts = []
        for jwst_target in programs:
            host_names = [obs['target'] for obs in jwst_target]
            nea_host = np.unique([
                host_aliases[host] if host in host_aliases else host
                for host in host_names
            ])
            jwst_hosts += list(nea_host)
            for obs in jwst_target:
                obs['nea_host'] = nea_host
        jwst_hosts = np.unique(jwst_hosts)

        planet_aliases = load_aliases('planet')
        planets_aka = u.invert_aliases(planet_aliases)

        for target in self.targets:
            target.is_jwst_host = target.host in jwst_hosts
            if target.is_jwst_host:
                for j in range(njwst):
                    if target.host == programs[j][0]['nea_host']:
                        break
                target.programs = programs[j]
                planets = []
                for obs in programs[j]:
                    planets += obs['planets']
                planets = np.unique(planets)
                letter = u.get_letter(target.planet).strip()
                target.is_jwst_planet = letter in planets
            else:
                target.is_jwst_planet = False

            if target.planet in planets_aka:
                target.aliases = planets_aka[target.planet]

        self._transit_mask = [target.is_transiting for target in self.targets]
        self._jwst_mask = [target.is_jwst_host for target in self.targets]
        self._confirmed_mask = [target.is_confirmed for target in self.targets]


    def get_target(
            self, name=None,
            is_transit=True, is_jwst=None, is_confirmed=None,
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
    database: String
        nea_data.txt or tess_data.txt

    Returns
    -------
    targets: List of Target

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> nea_data = cat.load_nea_targets_table()
    """
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


def _add_planet_info(observations):
    """
    Add planet letter info to a list of observations
    and other corrections
    """
    planets_file = f'{ROOT}data/programs/planets_per_program.json'
    with open(planets_file, "r") as f:
        planet_data = json.load(f)

    known_obs = []
    for obs in observations:
        pid = obs['pid']
        obs_id = obs['observation']
        key = f'{pid}_{obs_id}'
        known_obs.append(key)
        if key in planet_data:
            for var, value in planet_data[key].items():
                obs[var] = value

    for key,obs in planet_data.items():
        if key not in known_obs and 'missing' in obs:
            obs.pop('missing')
            date_format = "%Y-%m-%d %H:%M:%S"
            val = obs['date_start']
            if isinstance(val, str):
                obs['date_start'] = datetime.strptime(val, date_format)
            val = obs['date_end']
            if isinstance(val, str):
                obs['date_end'] = datetime.strptime(val, date_format)
            observations.append(obs)

    return observations


def _group_by_target(observations):
    """
    Group observations by host, using RA and dec to detect aliases
    for a same object
    """
    ra = [Angle(obs['ra'], unit=hourangle).deg for obs in observations]
    dec = [Angle(obs['dec'], unit=deg).deg for obs in observations]
    coords = SkyCoord(ra, dec, unit='deg', frame='icrs')

    nobs = len(observations)
    taken = np.zeros(nobs, bool)
    group_indices = []
    for i in range(nobs):
        if taken[i]:
            continue
        seps = coords[i].separation(coords).to('arcsec').value
        indices = np.where(seps < 50)[0]
        taken[indices] = True
        group_indices.append(indices)

    grouped_data = []
    for i,indices in enumerate(group_indices):
        target = [observations[j] for j in indices]
        grouped_data.append(target)

    return grouped_data


def load_trexolists(grouped=False, trexo_file=None):
    """
    Extract the JWST programs' data from a trexolists.csv file.
    Note that trexolists know targets by their host star, not by
    individual planets in a given system.

    Parameters
    ----------
    grouped: Bool
        - If False, return a single 1D list with observations.
        - If True, return a nested list of the observations per target.
    trexo_file: String
        If None, extract data from default Gen TSO location.
        Otherwise, a path to a trexolists.csv file.

    Returns
    -------
    observations: 1D or 2D list of dictionaries
        A list of all JWST observations, where each item is a dictionary
        containing the observation's details.
        If grouped is True, the output is a nested list, where the
        observations are grouped per target (host).

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>>
    >>> # Get data as lists of individual programs:
    >>> observations = cat.load_trexolists()
    >>> # Show one of them:
    >>> observations[98]
    {'category': 'ERS',
     'pi': 'Batalha',
     'pid': '1366',
     'proprietary_period': 0,
     'target': 'WASP-39',
     'target_in_program': 'WASP-39',
     'observation': '3',
     'visit': '1',
     'status': 'Archived',
     'ra': '14:29:18.3955',
     'dec': '-03:26:40.20',
     'event': 'transit',
     'instrument': 'NIRSPEC',
     'mode': 'BOTS',
     'disperser': 'G395H',
     'filter': 'F290LP',
     'subarray': 'SUB2048',
     'readout': 'NRSRAPID',
     'groups': 70,
     'phase_start': 0.95248,
     'phase_end': 0.96275,
     'duration': 10.56,
     'plan_window': None,
     'date_start': datetime.datetime(2022, 7, 30, 20, 46, 32),
     'date_end': datetime.datetime(2022, 7, 31, 6, 21, 30),
     'planets': ['b']}

    >>> # Get data grouped per target (host):
    >>> observations = cat.load_trexolists(grouped=True)
    >>> for obs in observations[27]:
    >>>     print(f"{obs['pid']}  {obs['pi']}  {obs['instrument']} {obs['event']} ")
    1366  Batalha  NIRISS transit
    1366  Batalha  NIRCAM transit
    1366  Batalha  NIRSPEC transit
    1366  Batalha  NIRSPEC transit
    2783  Powell  MIRI transit
    5634  Baeyens  NIRSPEC see_phase
    """
    if trexo_file is None:
        trexo_file = f'{ROOT}data/trexolists.csv'

    trexolist_data = ascii.read(
        trexo_file,
        format='csv', guess=False, fast_reader=False, comment='#',
    )

    nirspec_filter = {
        'G395H': 'F290LP',
        'G395M': 'F290LP',
        'G235H': 'F170LP',
        'G235M': 'F170LP',
        'G140H': 'F100LP',
        'G140M': 'F100LP',
        'PRISM': 'CLEAR',
    }
    instrument = {
        'BOTS': 'NIRSPEC',
        'SOSS': 'NIRISS',
        'GTS': 'NIRCAM',
        'LRS': 'MIRI',
        'MRS': 'MIRI',
        'F1500W': 'MIRI',
        'F1280W': 'MIRI',
    }

    observations = []
    for i,data in enumerate(trexolist_data):
        obs = {}
        obs['category'] = str(data['ProposalCategory'])
        obs['pi'] = str(data['LastName'])
        obs['pid'] = str(data['ProposalID'])
        obs['cycle'] = str(data['Cycle'])
        obs['proprietary_period'] = int(data['ProprietaryPeriod'])

        target = str(data['hostname_nn'])
        obs['target'] = u.normalize_name(target)
        obs['target_in_program'] = target
        obs['planets'] = data['letter_nn'].split('+')
        obs['event'] = data['Event'].lower().replace('phasec', 'phase curve')

        obs['observation'] = str(data['Observation'])
        obs['visit'] = '1'
        obs['status'] = str(data['Status'])
        coordinates = data['EquatorialCoordinates'].split()
        obs['ra'] = ':'.join(coordinates[0:3])
        obs['dec'] = ':'.join(coordinates[3:6])

        mode = str(data['ObservingMode'])
        disperser = obs['disperser'] = str(data['GratingGrism'])
        inst = obs['instrument'] = instrument[mode]
        if mode == 'SOSS':
            disperser = 'None'
            filter = 'CLEAR'
        elif mode == 'LRS':
            disperser = 'None'
            filter = 'None'
        elif mode == 'MRS':
            disperser = 'unknown'
            filter = 'None'
        elif inst == 'MIRI':
            # disperser will be fixed below by _add_planet_info()
            disperser = 'None'
            filter = mode
            mode = 'Imaging TS'
        elif mode == 'GTS':
            mode = 'GRISMR TS'
            disperser, filter = disperser.split('+')
            if '_' in data['Subarray']:
                disperser = f'DHS0,{disperser}'
                # hard-coded, known up to Cycle4:
                # will be fixed below by _add_planet_info()
                filter = f'F150W2,{filter}'
        elif inst == 'NIRSPEC':
            filter = nirspec_filter[disperser]
        obs['mode'] = mode
        obs['disperser'] = disperser
        obs['filter'] = filter

        obs['subarray'] = str(data['Subarray'])
        obs['readout'] = str(data['ReadoutPattern'])
        obs['groups'] = int(data['Groups'])

        window = str(data['PlanWindow'])
        if window == 'X':
            obs['plan_window'] = None
        elif '(' in window:
            window = window[0:window.index('(')]
            w_start, w_end = window.split('-')
            start = datetime.strptime(w_start.strip(), '%b %d, %Y')
            end = datetime.strptime(w_end.strip(), '%b %d, %Y')
            obs['plan_window'] = f"{start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}"
        else:
            obs['plan_window'] = window

        obs['duration'] = float(data['Hours'])

        date = data['StartTime']
        if date == 'X':
            obs['date_start'] = None
        else:
            obs['date_start'] = datetime.strptime(date, '%b %d, %Y %H:%M:%S')
        date = data['EndTime']
        if date == 'X':
            obs['date_end'] = None
        else:
            obs['date_end'] = datetime.strptime(date, '%b %d, %Y %H:%M:%S')

        observations.append(obs)

    observations = _add_planet_info(observations)

    if grouped:
        return _group_by_target(observations)
    return observations


def load_programs(grouped=False, csv_file=None):
    """
    Get the data from the downloaded JWST programs (xml files)
    Note that the programs know targets by host star, not by
    individual planets in a given system.

    Parameters
    ----------
    grouped: Bool
        - If False, return a 1D list of all observations
        - If True, return a nested list of observations grouped by
          host target.
    csv_file: String
        Path to a csv file saved with parse_programs().
        If None, load the default csv file of gen_tso (which should
        contain all known programs).

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> programs = cat.load_programs()
    """
    # Read the CSV file into a list of dictionaries
    if csv_file is None:
        csv_file = f'{ROOT}data/programs/jwst_tso_programs.csv'

    observations = []
    with open(csv_file, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            observations.append(row)

    # Parse data types
    int_keys = ['cycle', 'groups', 'integrations', 'proprietary_period']
    float_keys = ['duration', 'period', 'phase_start', 'phase_duration']
    date_keys = ['date_start', 'date_end']
    for obs in observations:
        for key,val in obs.items():
            if val == '':
                obs[key] = None
            elif key in int_keys:
                obs[key] = int(val)
            elif key in float_keys:
                obs[key] = float(val)
            elif key in date_keys:
                date_format = "%Y-%m-%d %H:%M:%S"
                obs[key] = datetime.strptime(val, date_format)
            elif key == 'planets':
                obs[key] = eval(obs[key])

    if grouped:
        return _group_by_target(observations)
    return observations


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

