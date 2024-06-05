# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'find_target',
    'Catalog',
    'load_trexolists',
    'load_targets',
    'load_aliases',
]

from astropy.io import ascii
import numpy as np
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

from ..utils import ROOT
from . import utils as u
from .target import Target


def find_target(targets=None):
    """
    Interactive prompt with tab-completion to search for targets.

    Parameters
    ----------
    targets: list of Target objects
    """
    if targets is None:
        targets = load_targets('nea_data.txt', is_confirmed=True)
    planets = [target.planet for target in targets]
    aliases = []
    for target in targets:
        aliases += target.aliases
    planets += list(aliases)

    completer = WordCompleter(
        planets,
        sentence=True,
        match_middle=True,
    )
    name = prompt(
        'Enter Planet name: ',
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

        self.planets = (
            [target.planet for target in nea_targets] +
            [target.planet for target in tess_targets]
        )

        # JWST targets
        jwst_hosts, trexo_ra, trexo_dec = load_trexolists(extract='coords')
        njwst = len(jwst_hosts)
        host_aliases = load_aliases(as_hosts=True)
        #hosts = [target.host for target in self.targets]
        #hosts_aka = u.invert_aliases(host_aliases)
        for i in range(njwst):
            if jwst_hosts[i] in host_aliases:
                jwst_hosts[i] = host_aliases[jwst_hosts[i]]
            # if host NEA name != planet NEA name
            #if jwst_hosts[i] not in hosts:
            #    print(jwst_hosts[i])
            #    for host in hosts_aka[jwst_hosts[i]]:
            #        if host in self.hosts:
            #            jwst_hosts[i] = host

        planet_aliases = load_aliases()
        planets_aka = u.invert_aliases(planet_aliases)

        for target in self.targets:
            target.is_jwst = target.host in jwst_hosts and target.is_transiting
            if target.is_jwst:
                j = list(jwst_hosts).index(target.host)
                target.trexo_ra_dec = (trexo_ra[j], trexo_dec[j])

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
            transit_dur, rplanet, mplanet, sma, period, teq = planet_vals

            target = Target(
                host=host,
                mstar=mstar, rstar=rstar, teff=teff, logg_star=logg,
                metal_star=metal,
                ks_mag=ks_mag, ra=ra, dec=dec,
                planet=planet,
                mplanet=mplanet, rplanet=rplanet,
                period=period, sma=sma, transit_dur=transit_dur,
                is_confirmed=is_confirmed,
            )
            targets.append(target)

    return targets


def load_trexolists(grouped=False, trexo_file=None, extract='target'):
    """
    Get the target names from the trexolists.csv file.

    Parameters
    ----------
    grouped: Bool
        If False, return a 1D list of names.
        If True, return a nested list of lists, with each item the
        set of names for a same object.
    trexo_list: String
        If None, extract data from default Gen TSO location.
        Otherwise, a path to a trexolists.csv file.
    extract: String
        If 'target' extract only the target names.
        If 'coords' extract the RA and dec in addition to the target names.
        Note that these coordinates are intentionally truncated so
        that a same object has a unique RA,dec (and thus can be used
        to identify the same target with the trexolists filters).
    """
    if trexo_file is None:
        trexo_file = f'{ROOT}data/trexolists.csv'

    trexolist_data = ascii.read(
        trexo_file,
        format='csv', guess=False, fast_reader=False, comment='#',
    )

    targets = trexolist_data['Target'].data
    norm_targets = [
        u.normalize_name(target)
        for target in targets
    ]

    ra = trexolist_data['R.A. 2000'].data
    dec = trexolist_data['Dec. 2000'].data
    truncated_ra = np.array([r[0:7] for r in ra])
    truncated_dec = np.array([d[0:6] for d in dec])

    if not grouped:
        unique_targets, u_idx = np.unique(norm_targets, return_index=True)
        if extract == 'coords':
            trexo_ra = truncated_ra[u_idx]
            trexo_dec = truncated_dec[u_idx]
            return unique_targets, trexo_ra, trexo_dec
        return unique_targets

    # Use RA and dec to detect aliases for a same object
    target_sets = []
    trexo_ra = []
    trexo_dec = []
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
        target_sets.append(list(np.unique(hosts)))
        trexo_ra.append(ra)
        trexo_dec.append(dec)

    if extract == 'coords':
        return target_sets, trexo_ra, trexo_dec
    return target_sets


def load_aliases(as_hosts=False):
    """
    Load file with known aliases of NEA targets.
    """
    with open(f'{ROOT}data/target_aliases.txt', 'r') as f:
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
            if parse(alias) != name:
                aliases[parse(alias)] = name
    return aliases

