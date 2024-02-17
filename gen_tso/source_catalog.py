import multiprocessing as mp
import pickle
import re
import urllib
import warnings

from astropy.io import ascii
import numpy as np
from astroquery.simbad import Simbad as simbad
#from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive as nasa
import requests
#import bs4
from bs4 import BeautifulSoup

"""
- fetch ps database
- fetch tess database
- fetch aliases
- scrap tess candidates info
- build database
"""

def fetch_database():
    """
    Fetch the entire NASA Exoplanet Archive database

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


    # TOI candidates: DOES NOT WORK
    #r = requests.get(
    #    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    #    "select+hostname,pl_name,default_flag,sy_kmag,sy_pnum,disc_facility,"
    #    "st_teff,st_logg,st_met,st_rad,st_mass,st_age,pl_trandur,"
    #    "pl_orbper,pl_orbsmax,pl_rade,pl_masse,pl_ratdor,pl_ratror+"
    #    "from+ps+"
    #    "where+hostname=%27TOI-260%27"
    #    #"+or+hostname=%27TOI-260%27"
    #    "&format=json"
    #)
    #if not r.ok:
    #    raise ValueError("Something's not OK")
    ##print(r.text)
    #r.json()


    # TOI candidates:
    # download table manually from browser :(
    candidates_csv = 'data/TOI_2023.09.02_12.15.44.csv'
    table = ascii.read(
        candidates_csv, format='csv',
        guess=False, fast_reader=False, comment='#',
    )

    tfop = table['tfopwg_disp'].data.data
    tfop[tfop=='0'] = ''
    teff = table['st_teff'].data.data
    logg = table['st_logg'].data.data
    st_rad = table['st_rad'].data.data
    pl_rad = table['pl_rade'].data.data
    period = table['pl_orbper'].data.data

    candidates = []
    for i,target in enumerate(table):
        candidate = {
            'pl_name': 'TOI-' + str(target['toi']),
            'hostname': 'TOI-' + str(target['toipfx']),
            'st_tmag': target['st_tmag'],
            'tfop': tfop[i],
            'st_teff': teff[i],
            'st_logg': logg[i],
            'st_rad': st_rad[i],
            'pl_trandur': target['pl_trandurh'],
            'pl_orbper': period[i],
            'pl_rade': pl_rad[i],
            'pl_ratror': np.sqrt(target['pl_trandep']*1e-6),
            #'default_flag': 1,
            #'sy_kmag': None,
            #'disc_facility': 'Xinglong Station',
            #'st_met': None,
            #'st_mass': None,
            #'st_age': None,
            #'pl_orbsmax': target[''],
            #'pl_masse': None,
            #'pl_ratdor': None,
        }
        candidates.append(candidate)

    # Need to scrap
    # sy_kmag, st_met (gaia), pl_masse, st_mass, pl_orbsmax


    for i,candidate in enumerate(candidates):
        aliases, kmag = simbad_aliases(candidate['hostname'], verbose=False)
        if kmag is None:
            print(f"no Simbad for {candidate['tfop']} target {candidate['hostname']}")
        candidate['sy_kmag'] = kmag
        if i%250 == 0:
            print(f'{i}/{len(candidates)}')

    # Keep APC, KP, PC and simbad finds
    tfops = []
    for candidate in candidates:
        keeper = (
            candidate['tfop'] in ['PC', 'KP', 'APC'] or
            candidate['tfop']=='' and candidate['sy_kmag'] is not None
        )
        if keeper:
            tfops.append(candidate['tfop'])
    utfop, counts = np.unique(tfops, return_counts=True)
    print(np.sum(counts))

    with open('data/nea_tess_candidates.pickle', 'wb') as handle:
        pickle.dump(candidates, handle, protocol=pickle.HIGHEST_PROTOCOL)


def tess_scrapper(target):
    """
    Need to scrap
    + sy_kmag
    - st_met (gaia)
    + st_mass
    - pl_masse
    - pl_orbsmax
    """
    target = 'TOI-260'
    target = 'TOI-905'
    target = 'TOI-904'
    response = requests.get(
        url=f'https://exoplanetarchive.ipac.caltech.edu/overview/{target}',
    )
    soup = BeautifulSoup(response.content, 'html.parser')
    pm = '±'

    kmag = None
    # Scrap the info:
    for dd in soup.find_all('dd'):
        texts = dd.get_text().split()
        if 'mKs' in texts:
            print(dd.text.split())
            kmag_err = texts[-1]
            kmag = float(kmag_err[0:kmag_err.find(pm)])

    planet_divs = [
        div for div in soup.find_all('div')
        if div.has_attr('id')
        if div['id'].startswith('planet_data_TOI-')
    ]
    nplanets = len(planet_divs)

    metal = None
    mstar = None
    planets = []
    mplanet = [None for _ in range(nplanets)]
    smaxis = [None for _ in range(nplanets)]
    fields =  ['Metallicity', 'M✶', '(M⨁)', '(au)', 'P']
    for i in range(nplanets):
        name = planet_divs[i]['id'][12:]
        planets.append(f'{name[0:-3]}.{name[-2:]}')
        for tr in planet_divs[i].find_all('tr'):
            texts = tr.get_text().split()
            intersect = np.intersect1d(texts, fields)
            if len(intersect) > 0:
                #print(texts)
                for solution in texts[2:]:
                    if solution == '---':
                        continue

                    if pm in solution:
                        pm_idx = solution.index(pm)
                    elif '+' in solution:
                        pm_idx = solution.index('+')
                    else:
                        continue
                    val = float(solution[0:pm_idx])
                    #print(val)
                    if 'M✶' in texts and mstar is None:
                        mstar = val
                    if 'Metallicity' in texts and metal is None:
                        metal = val
                    if '(M⨁)' in texts and mplanet[i] is None:
                        mplanet = val
                    if '(au)' in texts and smaxis[i] is None:
                        smaxis = val
        print(kmag, metal, mstar, planets, mplanet, smaxis)
    return kmag, metal, mstar, mplanet, smaxis


def load_nea_table():
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
    with open('data/nea_tess_candidates.pickle', 'rb') as handle:
        tess = pickle.load(handle)
    return tess



def simbad_aliases(target, verbose=True):
    #simbad.list_votable_fields()
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
    #if False:
    #    for var,val in simbad_info.items():
    #        print('\n', val)

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


def get_nea_aliases(target):
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


def fetch_aliases():
    """
    Wait, what?:  'HAT-P-70'  'HAT-P-70b'
    """
    nea_data = load_nea_table()
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
            new_aliases = pool.map(get_nea_aliases, hosts[first:last])

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
            new_aliases = pool.map(get_nea_aliases, toi_hosts[first:last])

        for new_alias in new_aliases:
            aliases.update(new_alias)
        print(f'{last} / {nhosts}')

    with open('data/nea_all_aliases.pickle', 'wb') as handle:
        pickle.dump(aliases, handle, protocol=pickle.HIGHEST_PROTOCOL)


    #with open('data/nea_all_aliases.pickle', 'rb') as handle:
    #    aliases = pickle.load(handle)

    hosts = np.unique(list(nea_data[1]) + toi_hosts)
    nea_names = np.unique(list(aliases.values()))
    for target in hosts:
        #target = hosts[4009]
        s_aliases, kmag = simbad_aliases(target)
        new_aliases = []
        for alias in s_aliases:
            alias = re.sub('\s+', ' ', alias)
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


def scrap_data():
    with open('data/nea_all_aliases.pickle', 'rb') as handle:
        aliases = pickle.load(handle)

    with open('data/nea_data.pickle', 'rb') as handle:
        nea = pickle.load(handle)
    nea_hosts = [entry['hostname'] for entry in nea]

    nea_data = load_nea_table()
    tess = load_nea_tess_table()

    hosts = np.unique(nea_data[1])
    nhosts = len(hosts)

    # Add TESS candidates
    candidates = []
    for candidate in tess:
        keeper = (
            candidate['tfop'] in ['PC', 'KP', 'APC']
        )
        if keeper:
            candidates.append(candidate)
    tess_hosts = [target['hostname'] for target in candidates]

    candidate_hosts = []
    for i in range(len(tess_hosts)):
        if tess_hosts[i] in aliases:
            host = aliases[tess_hosts[i]]
        else:
            host = tess_hosts[i]
        if host not in hosts:
            candidate_hosts.append(host)

    for host in candidate_hosts:
        planets = [
            planet['pl_name']
            for planet in tess
            if planet['hostname'] == host
        ]
        print(planets)


def build_database():
    """
    one entry per system
    """
    with open('data/nea_all_aliases.pickle', 'rb') as handle:
        aliases = pickle.load(handle)

    with open('data/nea_data.pickle', 'rb') as handle:
        nea = pickle.load(handle)
    nea_hosts = [entry['hostname'] for entry in nea]

    nea_data = load_nea_table()
    tess = load_nea_tess_table()

    hosts = np.unique(nea_data[1])
    nhosts = len(hosts)

    # Add TESS candidates
    candidates = []
    for candidate in tess:
        keeper = (
            candidate['tfop'] in ['PC', 'KP', 'APC']
        )
        if keeper:
            candidates.append(candidate)
    tess_hosts = [target['hostname'] for target in candidates]

    candidate_hosts = []
    known_hosts = []
    for i in range(len(tess_hosts)):
        if tess_hosts[i] in aliases:
            host = aliases[tess_hosts[i]]
        else:
            host = tess_hosts[i]
        if host not in hosts:
            candidate_hosts.append(host)
        else:
            known_hosts.append(host)

    all_hosts = np.concatenate((hosts,candidate_hosts))

    # Unique objects (hosts and planets)
    targets = np.unique(list(aliases.values()))
    is_host = [target in all_hosts for target in targets]

    aka = get_aka(aliases)

    for host in all_hosts:
        # host = all_hosts[3992]  # WASP-47
        host = 'TOI-260'
        if host in nea_hosts:
            entries = nea
        else:
            entries = tess
        planets = [
            entry
            for entry in entries
            if entry['hostname'] == host
        ]
        # if host in nea
        data = planets[0]
        system = {}
        system['host'] = host
        system['aliases'] = aka[host]
        system['teff'] = data['st_teff']
        system['logg'] = data['st_logg']
        system['kmag'] = data['sy_kmag']
        system['rstar'] = data['st_rad']
        system['metal'] = data['st_met']
        system['planets'] = []
        for data in planets:
            planet = {}
            planet['name'] = data['pl_name']
            planet['radius'] = data['pl_rade']
            planet['mass'] = data['pl_masse']
            planet['smaxis'] = data['pl_orbsmax']
            planet['T14'] = data['pl_trandur']
            planet['aliases'] = aka[data['pl_name']]
            #planet[''] = 
            system['planets'].append(planet)

    #absolutely need:
    #- st_rad (tsm, esm)   or [pl_ratror]
    #- pl_orbsmax (tsm, esm)  or [pl_ratdor] or [pl_orbper and st_mass]


def clean(word, chars='- '):
    """
    Remove white spaces and dashes when possible withot creating
    ambiguities (e.g., not in between two digits)
    """
    nchars = len(word)
    for i in range(1, nchars):
        # Found char
        remove = word[i] in chars
        # Not surrounded by digits
        remove &= (
            i == nchars-1 or
            not (word[i-1].isdigit() and word[i+1].isdigit())
        )
        # Except this is a BD catalog
        if i == 2 and word.startswith('bd'):
            remove = False

        # Remove character and call recursively on new word
        if remove:
            word = word[0:i] + word[i+1:]
            #print(word)
            return clean(word, chars)
    return word


def trexolist_crosscheck():
    nea_data = load_nea_table()
    hosts = nea_data[1]
    with open('data/nea_all_aliases.pickle', 'rb') as handle:
        aliases = pickle.load(handle)


    jwst = ascii.read(
        'data/trexolists.csv',
        format='csv', guess=False, fast_reader=False, comment='#',
    )
    targets = jwst['Target'].data
    trix = np.unique(targets[np.in1d(targets, hosts, invert=True)])
    # Missing targets
    # 'GJ-436-offset'  False   None  # remove -offset
    # 'WD1856'         False   None  # WD 1856+534, append missing info?
    # 'RHO01-CNC'      False   None  # Remove leading zero?
    # 'GJ-4102A'       False   None  # TOI-910 (remove trailing A?)


def what_is_this():
    #matches = [alias.lower() for alias in sorted_aliases]
    matches = np.array([
        clean(alias.lower()).replace(' ', '-')
        for alias in aliases.keys()
    ])
    sort_idx = np.argsort(matches)
    matches = matches[sort_idx]
    sorted_aliases = np.array(list(aliases.keys()))[sort_idx]


def index_binsearch(target, target_list, first=0, last=None):
    """
    target_list = 'a c e g'.split()
    target = 'g'
    first = 0
    index_binsearch(target, target_list)
    """
    if last is None:
        last = len(target_list) 
    if last - first <= 1:
        if target_list[first] == target:
            return first
        return -1

    mid = (first + last) // 2
    if target < target_list[mid]:
        return index_binsearch(target, target_list, first=first, last=mid)
    else:
        return index_binsearch(target, target_list, first=mid, last=last)


def match_target(target, matches):
    """
    for target in targets:
    for target in trix:
        match = match_target(target, matches)
        print(f'{repr(target):21}  {match is not None}   {repr(match)}')
    """
    # Case-insensitive, no need to worry about stellar letters
    # NEA does not accept 'WASP-94 b' in place of 'WASP-94 B b'
    match_target = clean(target.strip().lower()).replace(' ', '-')

    # Fixes
    # Gliese
    if match_target.startswith('gl'):
        match_target = 'gj' + match_target[2:]
    # General Catalog of Variable
    if match_target.startswith('vv'):
        match_target = match_target[1:]

    # Needed to implement a binary search because standard 'in' was lagging
    index = index_binsearch(match_target, matches)
    if index == -1:
        return None
    match = sorted_aliases[index]
    return aliases[match]


def get_roots(matches, start_idx=0):
    # Next divergence
    str_size = np.amax(np.char.str_len(matches))
    chars = np.array([list(f.ljust(str_size)) for f in matches])
    for split_idx in range(start_idx, str_size):
        char_set = set(chars[:,split_idx])
        #print(k, sorted(set(chars[:,k])))
        if len(char_set) > 1:
            break
    return split_idx, sorted(char_set)


#lower_case = [entry.lower() for entry in planets]
def completions(text, lower_case):
    #value = self.ent_var.get().strip().lower()
    #text = 'ke'
    value = text.lower()
    input_len = len(value)
    matches = np.array([f for f in lower_case if f.startswith(value)])
    # Next divergence
    split_idx, char_set = get_roots(matches, input_len)

    # auto-suggest
    root = matches[0][0:split_idx]
    suggestion = None
    if split_idx > input_len:
        suggestion = root

    print(
        f'word: {repr(text)}  [{input_len}, {split_idx}]\n'
        f'Auto-suggest: {split_idx>input_len}\n'
        f'Next chars: {char_set}'
    )

    # Get list of completions
    completions = []
    for char in char_set:
        #print(root + char)
        value = (root + char).lower()
        input_len = len(value)
        c_matches = np.array([f for f in lower_case if f.startswith(value)])
        if len(c_matches) == 1:
            completions.append(c_matches[0])
        else:
            # Next divergence
            c_split_idx, char_set = get_roots(c_matches, input_len)
            completions.append(c_matches[0][0:c_split_idx])
    #print(completions)

    return suggestion, completions


def get_aka(aliases):
    aka = {}
    current_target = None
    for alias, target in aliases.items():
        if current_target is None:
            current_target = target
            target_aliases = [alias]
            continue
        if target != current_target:
            aka[current_target] = target_aliases
            current_target = target
            target_aliases = [alias]
        else:
            target_aliases.append(alias)
    return aka


def playground():
    for host in toi_hosts:
        if host not in aliases:
            print(f'missing: {host}')
        elif aliases[host] != host:
            print(host, aliases[host])
    # TOI-1241 KOI-5
    # missing: TOI-1259
    # TOI-2410 EPIC 220198551
    # TOI-2425 EPIC 220192485


    sorted_aliases = {
        k: v for k, v in sorted(aliases.items(), key=lambda item: item[1])
    }
    aka = {}
    current_target = None
    for alias, target in sorted_aliases.items():
        if current_target is None:
            current_target = target
            target_aliases = [alias]
            continue
        if target != current_target:
            aka[current_target] = target_aliases
            current_target = target
            target_aliases = [alias]
        else:
            target_aliases.append(alias)


    planets = np.unique(nea_data[0])
    nea_targets = np.unique(list(aliases.values()))

    # New targets
    for planet in nea_names:
        if planet not in nea_targets:
                print(planet)
    # Many are candidates, add their info?
    # TBD: remove 'B' and 'Ab' host companions (have no info)
    # TBD: remove false positives

    for planet in nea_names:
        if planet not in hosts:
            if planet[-1].isalpha() and not planet[-2].isspace():
                print(planet)

