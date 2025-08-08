# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    # To fetch the JWST programs info from the internets
    '_extract_html_text',
    '_fetch_programs',
    '_extract_instrument_template',
    'fetch_jwst_programs',
    'update_jwst_programs',
    # To format XML info into human language
    '_parse_date',
    '_parse_window',
    '_get_phase_info',
    'guess_event_type',
    'parse_status',
    'parse_program',
    # To guess the planet in an observation
    '_planet_from_label',
    '_planet_from_period',
    '_clean_label',
    'get_observation_host',
    'get_planet_letters',
]

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import os
import re
import shutil
import warnings
import xml.etree.ElementTree as ET

import numpy as np
from bs4 import BeautifulSoup
import pyratbay.constants as pc
import requests

from ..utils import ROOT, KNOWN_PROGRAMS
from .utils import (
    normalize_name,
    get_host,
    get_letter,
)
from .catalogs import load_programs, Catalog


def _extract_html_text(html):
    """
    Parse an HTML file, extract content inside its <main> tag
    """
    soup = BeautifulSoup(html, "html.parser")
    main_tag = soup.find("main")
    return main_tag.get_text(separator=" ", strip=True) if main_tag else None


def _fetch_programs(programs, output_path, file_type='APT'):
    """
    Fetch APT files or status XML files from the Space Telescope website.

    Parameters
    ----------
    programs: 1D integer iterable.
        Program IDs.
    output_path: String
        Output folder location.
    file_type: String
        'APT' or 'status'.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> programs = [1366, 1729, 1981, 2055, 9999]
    >>> output_path = f'{ROOT}data/programs/'
    >>> file_type = 'APT'
    >>> cat._fetch_programs(programs, output_path, file_type)
    """
    nprograms = len(programs)

    if file_type == 'APT':
        urls = np.array([
            f'https://www.stsci.edu/jwst-program-info/download/jwst/apt/{pid}/'
            for pid in programs
        ])
    elif file_type == 'status':
        urls = np.array([
            f'https://www.stsci.edu/jwst-program-info/visits/?program={pid}&download=&pi=1'
            for pid in programs
        ])

    def fetch_url(url):
        try:
            response = requests.get(url)
            return response
        except:
            return None

    fetch_status = np.tile(2, nprograms)
    responses = np.tile({}, nprograms)
    n_attempts = 0
    while np.any(fetch_status>0) and n_attempts < 10:
        n_attempts += 1
        mask = fetch_status > 0
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(fetch_url, urls[mask]))

        j = 0
        for i in range(nprograms):
            if fetch_status[i] <= 0:
                continue
            r = results[j]
            j += 1
            if r is None:
                continue
            if not r.ok:
                warnings.warn(
                    f"{file_type} fetch failed for program {programs[i]} "
                    f"with error {r.status_code}: {repr(r.reason)}"
                )
                fetch_status[i] = -1
                continue
            if file_type=='status' and 'xml' not in r.headers['Content-Type']:
                warnings.warn(_extract_html_text(r.text))
                fetch_status[i] = -1
                continue
            responses[i] = r.content
            fetch_status[i] = 0
        fetched = np.sum(fetch_status <= 0)
        print(f'Fetched {fetched}/{nprograms} {file_type} files on try {n_attempts}')

    for i,resp in enumerate(responses):
        if fetch_status[i] != 0:
            continue
        pid = programs[i]
        if file_type == 'APT':
            file_path = f'{output_path}/{pid}.aptx'
        elif file_type == 'status':
            file_path = f'{output_path}/JWST-{pid}-visit-status.xml'
        with open(file_path, mode="wb") as file:
            file.write(resp)
    return np.array(programs)[fetch_status==0]


def fetch_jwst_programs(programs, apt_command=None, output_path=None):
    r"""
    Fetch APT and status for requested programs.
    APT files will be converted to XML using apt_command.

    Parameters
    ----------
    programs: integer or 1D iterable of intergers
        The program PID numbers to fetch.
    apt_command: String
        APT's command executable.
        If None, do not convert .aptx files to .xml
    output_path: String
        Output folder location.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> from gen_tso.utils import KNOWN_PROGRAMS
    >>>
    >>> # Fetch all known TSO programs
    >>> apt_command = '/Applications/APT\\ 2025.1/bin/apt'
    >>> programs = KNOWN_PROGRAMS
    >>> cat.fetch_jwst_programs(programs, apt_command)
    >>>
    >>> # Fetch a specific TSO program
    >>> programs = 1366
    >>> cat.fetch_jwst_programs(programs, apt_command)
    """
    if isinstance(programs, str) or isinstance(programs, int):
        programs = [programs]

    if output_path is None:
        output_path = f'{ROOT}data/programs/'

    # Fetch status xml files
    _fetch_programs(programs, output_path, 'status')

    # Fetch APT files, export to xml
    apt_programs = _fetch_programs(programs, output_path, 'APT')
    if apt_command is not None:
        if apt_command in os.environ:
           apt_command = os.environ[apt_command]
        apt_files = " ".join(
            [f'{output_path}/{pid}.aptx' for pid in apt_programs]
        )
        os.system(f"{apt_command} -nogui -export xml {apt_files}")


def update_jwst_programs():
    """
    Update JWST programs, re-processing only the necessary APTs
    """
    output_path = f'{ROOT}data/programs/'
    programs = KNOWN_PROGRAMS
    to_csv = f'{ROOT}data/programs/jwst_tso_programs.csv'
    failed = ['Failed', 'Skipped', 'Withdrawn']

    if 'APT' not in os.environ:
        # Only update status
        _fetch_programs(programs, output_path, 'status')
    else:
        # Fetch status in tmp folder
        apt_command = os.environ['APT']
        tmp_path = f'{ROOT}data/programs_tmp/'
        _fetch_programs(programs, tmp_path, 'status')

        # Find observations that failed since last update
        observations = load_programs()
        pid = ''
        to_update = []
        for obs in observations:
            if obs['pid'] != pid:
                pid = obs['pid']
                status = parse_status(pid, path=tmp_path)
            for i in range(len(status)):
                if status[i]['observation'] == obs['observation']:
                    break
            state = status[i]
            if state['status'] != obs['status'] and state['status'] in failed:
                to_update.append(pid)

        # Fetch APTs of programs with failed observations
        if len(to_update) > 0:
            fetch_jwst_programs(to_update, apt_command, output_path)
        # Move status xml files
        for file in os.listdir(tmp_path):
            if not file.endswith('-visit-status.xml'):
                continue
            src = os.path.join(tmp_path, file)
            dest = os.path.join(output_path, file)
            shutil.move(src, dest)

    # Parse all
    parse_program(programs, to_csv=to_csv)


def _extract_instrument_template(element):
    """
    Extract the instrumental settings from an observation's XML template.
    """
    modes = {
        'MiriImaging': 'Imaging TS',
        'MiriLRS': 'LRS',
        'MiriMRS': 'MRS',
        'NircamGrismTimeSeries': 'GRISMR TS',
        'NirissSoss': 'SOSS',
        'NirspecBrightObjectTimeSeries': 'BOTS',
    }

    tag = element.tag
    len_ns = tag.rindex('}') + 1
    xmlns = tag[0:len_ns]
    mode = tag[len_ns:]
    template = {
        'mode': modes[mode],
    }
    if template['mode'] == 'BOTS':
        template['disperser'] = element.find(f".//{xmlns}Grating").text
    elif template['mode'] == 'MRS':
        template['disperser'] = element.find(f".//{xmlns}Wavelength").text
    else:
        template['disperser'] = 'None'

    if template['mode'] == 'GRISMR TS':
        sw_filter = element.find(f".//{xmlns}ShortPupilFilter").text
        sw_disperser, sw_filter = sw_filter.split('+')
        sw_disperser = sw_disperser.replace('GDHS0', 'DHS0')
        lw_filter = element.find(f".//{xmlns}LongPupilFilter").text
        lw_disperser, lw_filter = lw_filter.split('+')
        template['disperser'] = f'{sw_disperser},{lw_disperser}'
        template['filter'] = f'{sw_filter},{lw_filter}'
    elif template['mode'] in ['Imaging TS', 'BOTS']:
        template['filter'] = element.find(f".//{xmlns}Filter").text
    elif template['mode'] in ['SOSS']:
        template['filter'] = 'CLEAR'
    else:
        template['filter'] = 'None'

    template['subarray'] = element.find(f".//{xmlns}Subarray").text
    if template['mode'] == 'MRS':
        readout = element.find(f".//{xmlns}ReadoutPatternLong").text
        groups = element.find(f".//{xmlns}GroupsLong").text
        integs = element.find(f".//{xmlns}IntegrationsLong").text
    else:
        readout = element.find(f".//{xmlns}ReadoutPattern").text
        groups = element.find(f".//{xmlns}Groups").text
        integs = element.find(f".//{xmlns}Integrations").text
    template['readout'] = readout
    template['groups'] = int(groups)
    template['integrations'] = int(integs)

    return template


def _parse_date(date_text):
    """
    Parse date format from a status' start/end date.

    Returns
    -------
    date:
        A datetime.datetime object
    """
    date_format = "%b %d, %Y %H:%M:%S"
    try:
        date = datetime.strptime(date_text, date_format)
    except:
        date = date_text
    return date


def _parse_window(window_text):
    """
    Parse date format from a status' start/end date.

    Returns
    -------
    window: String
        A string representation of the start - end date span
    """
    if "(" not in window_text:
        return "not yet assigned"

    idx_end = window_text.index('(')
    dates = window_text[0:idx_end].split('-')

    window_format = "%b %d, %Y"
    start = datetime.strptime(dates[0].strip(), window_format)
    end = datetime.strptime(dates[1].strip(), window_format)
    start_date = start.strftime('%Y-%m-%d')
    end_date = end.strftime('%Y-%m-%d')
    window = f"{start_date} - {end_date}"
    return window


def _get_phase_info(obs):
    """
    Extract the orbital period, starting phase, and duration
    from an observations from the PeriodZeroPhase constraints.

    Note that output values are adjusted to reflect the astrophysical
    values, since some constraints may have been modified to ease the
    scheduling.

    Parameters
    ----------
    obs: Dictionary
        A JWST observation dict.

    Returns
    -------
    period: Float
        Orbital period in days.
    phase: Float
        Orbital phase at start of observation (midpoint between
        PhaseStart and PhaseEnd).
    phase_duration: Float
        Duration of the observation in orbital phase units.
    """
    if obs['phase_reqs'] is None:
        return None, None, None
    phase_reqs = obs['phase_reqs']

    # The orbital period in days:
    value, unit = phase_reqs['Period'].split()
    per_unit = unit.lower().rstrip('s')
    period = float(value) * getattr(pc, per_unit) / pc.day

    duration = obs['duration']
    phase_duration = (duration*pc.hour) / (period*pc.day)

    # The mid-point of the starting phase range:
    phase_start = float(phase_reqs['PhaseStart'])
    phase_end = float(phase_reqs['PhaseEnd'])
    phase = 0.5*(phase_start+phase_end)
    if phase < 0:
         phase += 1

    # Phase curves have durations longer than 1 orbit,
    # but phase-curved have halved-period values to ease scheduling
    is_phase = phase_duration > 0.54
    phase_doubling = 1
    pid = obs['pid']
    if pid == '3860':
        phase_doubling = 4
    elif is_phase or pid in ['2084']:
        phase_doubling = 2

    phase_duration *= phase_doubling
    period /= phase_doubling
    phase *= phase_doubling
    # Manual guesstimate
    if pid == '1201' and 'eclipse' in obs['label']:
        phase -= 0.5
    phase -= int(phase)

    return period, phase, phase_duration


def guess_event_type(obs):
    """
    Use labels or phase constraints to guess whether the observation
    is an eclipse, transit, or phase curve.

    If there's orbital phase information, it will
    """
    pid = obs['pid']
    label = obs['label']
    event = ''

    # Guess from orbital phase when phase constraints exist:
    if obs['period'] is not None:
        phase = obs['phase_start']
        duration = obs['phase_duration']
        if duration > 1.0:
            event = 'phase curve'
        elif (phase<1.0) and (phase+duration>1.0):
            event = 'transit'
        else:
            event = 'eclipse'

    # Use label (can override phase-guess, and that's intentional):
    if 'trans' in label:
        event = 'transit'
    elif 'phase' in label:
        event = 'phase curve'
    elif 'eclipse' in label or 'emis' in label or 'occultat' in label:
        event = 'eclipse'

    # Hardcoded patches for missing information:
    if pid in ['2149', '2589', '3385', '5177', '5882', '6456']:
        event = 'transit'
    elif pid in ['2488', '2765']:
        event = 'phase curve'

    if event == '':
        obs_id = obs['observation']
        visit = obs['visit']
        print(f'{pid} {obs_id:3} {visit}  Could not identify event type')
    return event


def parse_status(pid, path=None):
    """
    Parse a program's status xml file to list of dictionaries

    Parameters
    ----------
    pid: Integer
        The programs PID number.
    path: String
        Folder where the status xml file is loacted.

    Returns
    -------
    visit_status: List of dictionaries
        Dictionaries with the observations' status for given program
    """
    if path is None:
        path = f'{ROOT}data/programs/'

    status = ET.parse(f'{path}/JWST-{pid}-visit-status.xml').getroot()
    visit_status = []
    for visit in status.findall('.//visit'):
        state = visit.attrib
        state['status'] = visit.find('status').text
        state['duration'] = float(visit.find('hours').text)
        target = visit.find('target')
        if target is None:
            target = 'None'
        else:
            target = target.text
        start = visit.find('startTime')
        end = visit.find('endTime')
        window = visit.find('planWindow')

        state['date_start'] = None
        state['date_end'] = None
        state['plan_window'] = None
        if start is not None:
            state['date_start'] = _parse_date(start.text)
        if end is not None:
            state['date_end'] = _parse_date(end.text)
        if window is not None:
            state['plan_window'] = _parse_window(window.text)
        visit_status.append(state)
    return visit_status


def parse_program(pid, path=None, to_csv=None):
    """
    Parse a JWST program's xml files (of APT and status) to extract
    the program's, status, and observation information.

    Parameters
    ----------
    pid: Integer or 1D list of integers.
        JWST TSO program ID number(s).
    path: String
        Path where the input XML files are located. These are
        files downloaded with cat.fetch_jwst_programs().
    to_csv: string
        If not None, save the outputs to a csv file located in
        given filename path.

    Returns
    -------
    observations: List of dictionaries
        Dictionaries with the observations information:
        - The program's category, PI, PID, cycle, and proprietary period
        - The target's name, ra, and dec
        - The observation's number, visit, duration (hours),
          status, start and end dates (or planned window), labels,
          and event type (eclipse, transit, or phase curve)
        - The instrument, detector mode, disperser, filter,
          subarray, readout, groups, and integrations.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> obs = cat.parse_program(pid=3712)

    >>> import gen_tso.catalogs as cat
    >>> obs = cat.parse_program(pid=3712, to_csv='jwst_tso_program_3712.csv')

    >>> import gen_tso.catalogs as cat
    >>> from gen_tso.utils import ROOT, KNOWN_PROGRAMS
    >>> # Save to default csv file
    >>> to_csv = f'{ROOT}data/programs/jwst_tso_programs.csv'
    >>> observations = cat.parse_program(pid=KNOWN_PROGRAMS, to_csv=to_csv)
    """
    if path is None:
        path = f'{ROOT}data/programs/'

    if isinstance(pid, (int, str)):
        pid = [pid]

    observations = []
    for program in pid:
        status = parse_status(program, path)
        # APT
        root = ET.parse(f'{path}/{program}.xml').getroot()
        ns = {'apt': 'http://www.stsci.edu/JWST/APT'}

        # The program
        pi_category = root.find('.//apt:ProposalCategory', ns).text
        cycle = root.find('.//apt:Cycle', ns).text
        prop = root.find('.//apt:ProprietaryPeriod', ns).text
        proprietary_period = prop[prop.find('[')+1:prop.find(' Month')]
        pi = ".//apt:PrincipalInvestigator/apt:InvestigatorAddress/apt:LastName"
        pi_lastname = root.find(pi, ns).text

        # The targets
        targets = {}
        for child in root.findall('.//apt:Targets/apt:Target', ns):
            target = {}
            target['number'] = child.find('apt:Number', ns).text
            target['ID'] = child.find('apt:TargetID', ns).text
            target['name'] = child.find('apt:TargetName', ns).text
            archive_name = child.find('apt:TargetArchiveName', ns)
            if archive_name is not None:
                target['archive_name'] = archive_name.text
            coords = child.find('apt:EquatorialCoordinates', ns)
            if coords is None:
                continue
            coords = coords.get('Value')
            target['ra'] = ':'.join(coords.split()[0:3])
            target['dec'] = ':'.join(coords.split()[3:6])
            epoch = child.find('apt:Epoch', ns)
            if epoch is not None:
                target['epoch'] = epoch.text
            targets[f"{target['number']} {target['ID']}"] = target

        # The observations
        for obs_group in root.findall( './/apt:ObservationGroup', ns):
            label = obs_group.find('apt:Label', ns)
            group_label = "" if label is None else label.text
            for obs in obs_group.findall('.//apt:Observation', ns):
                observation = {
                    'category': pi_category,
                    'pid': str(program),
                    'pi': pi_lastname,
                    'cycle': int(cycle),
                }
                reqs = np.unique([
                    child.tag[child.tag.rindex('}')+1:]
                    for child in obs.find(".//apt:SpecialRequirements", ns)
                ]).tolist()
                phase_reqs = obs.find(".//apt:PeriodZeroPhase", ns)
                time_series_reqs = obs.find(".//apt:TimeSeriesObservation", ns)
                if time_series_reqs is None:
                    continue

                observ = obs.find('apt:Number', ns).text
                visit = obs.find('apt:Visit', ns).get('Number')
                for state in status:
                    if state['observation'] == observ and state['visit'] == visit:
                        break
                observation['observation'] = state['observation']
                observation['visit'] = state['visit']
                observation['status'] = state['status']
                # Short observations are background:
                if state['duration'] < 1.5:
                    continue
                # placeholder
                observation['event'] = ''

                target_id = obs.find('apt:TargetID', ns).text
                target_name = targets[target_id]['name']
                norm_target = normalize_name(target_name)
                observation['target'] = norm_target
                observation['target_in_program'] = target_name

                observation['ra'] = targets[target_id]['ra']
                observation['dec'] = targets[target_id]['dec']
                observation['instrument'] = obs.find('apt:Instrument', ns).text
                template = obs.find('apt:Template', ns)[0]
                observation.update(_extract_instrument_template(template))

                observation['duration'] = state['duration']
                observation['date_start'] = state['date_start']
                observation['date_end'] = state['date_end']
                observation['plan_window'] = state['plan_window']
                observation['proprietary_period'] = int(proprietary_period)
                label = obs.find('apt:Label', ns)

                observation['label'] = group_label
                if label is not None:
                    label = label.text.lower()
                    observation['label'] = " : ".join([group_label, label])

                observation['special_reqs'] = reqs
                observation['phase_reqs'] = None
                if phase_reqs is not None:
                    observation['phase_reqs'] = phase_reqs.attrib
                # Add orbital-phase information when possible
                period, phase, obs_duration = _get_phase_info(observation)
                if period is None:
                    observation['period'] = None
                    observation['phase_start'] = None
                    observation['phase_duration'] = None
                else:
                    observation['period'] = period
                    observation['phase_start'] = phase
                    observation['phase_duration'] = obs_duration
                observation['event'] = guess_event_type(observation)
                observations.append(observation)

    # Cross check with NASA catalog to identify planets
    targets = [
        target for target in Catalog().targets
        if target.is_transiting
    ]
    for obs in observations:
        obs['planets'] = get_planet_letters(obs, targets)

    # Write to CSV file
    if to_csv is not None:
        fieldnames = {key for obs in observations for key in obs.keys()}
        fieldnames = []
        for obs in observations:
            for key in obs.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(to_csv, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in observations:
                writer.writerow(row)

    return observations


def _planet_from_label(text):
    """
    Extract planet letter from label by identifying single letters
    surrounded by spaces.

    Parameters
    ----------
    text: String
        The input string.

    Returns
    -------
    list: A list of extracted single letters.
    """
    # Edge-case, more than one planet:
    if '+' in text:
        label = text.replace(' + ', '+')
        idx = label.index('+')
        planet_letters = [label[idx-1], label[idx+1]]
        if planet_letters[0].isalpha() and planet_letters[1].isalpha():
            return planet_letters

    # Regular expression to match single letters surrounded by spaces
    pattern = r'(?<=\s)[a-zA-Z](?=\s)'
    # Find all matches
    matches = re.findall(pattern, f' {text} ')
    planet_letters = np.unique(matches).tolist()
    return planet_letters


def _planet_from_period(obs, planets):
    """
    Extract planet letter from known periods.
    """
    period = obs['period']
    periods = np.array([planet.period for planet in planets])
    idx = np.argmin(np.abs(periods-period))
    planet = planets[idx].planet
    letter = get_letter(planet).strip()
    return letter


def _clean_label(label, hosts):
    """
    Clean up an observation label. Mostly to make sure that the
    planet letter is isolated when running get_planet_letters().

    Parameters
    ----------
    label: String
    hosts: 1D iterable of strings

    Returns
    -------
    label: String
    """
    if label is None:
        return ''
    label = label.lower()
    label = label.replace('776.01', '776 c')
    label = label.replace('776.02', '776 b')
    label = label.replace('836.01', '836 c')
    label = label.replace('836.02', '836 b')
    label = label.replace('175.01', '175 c')
    label = label.replace('175.02', '836 b')
    label = label.replace('.01', ' b')
    label = label.replace('.02', ' c')

    label = label.replace('_transit', ' transit')
    label = label.replace('l23', 'l 23')
    label = label.replace('l98', 'l 98')
    label = label.replace('hd106', 'hd 106')
    label = label.replace('hip67', 'hip 67')

    for host in hosts:
        label = label.replace(host.lower(), '').strip()
    return label.strip()


def get_observation_host(obs, targets):
    """
    Cross-check observation's target with catalog to find the host name

    Parameters
    ----------
    observation: Dictionary
        A dict with a JWST TSO's information
    targets: List of Target
        Catalog of targets to cross-check the observation's target name.

    Returns
    -------
    host: String
        The host name of the observation's target.
    """
    target_name = obs['target']
    name = normalize_name(target_name)

    for target in targets:
        hosts = [target.host] + [get_host(alias) for alias in target.aliases]
        if name in hosts:
            return target.host
    # If not found
    return None


def get_planet_letters(obs, targets, verbose=False):
    """
    Find the planet(s) targeted by a given JWST observation
    These are found from the observations info by looking at:
    - if the target is in a single-planet system
    - if the target- or observation-label names the planet
    - if the period matches known orbital periods in the system
    - manually identified

    Parameters
    ----------
    observation: Dictionary
        A dict with a JWST TSO's information
    targets: List of Target
        Catalog of targets to cross-check the observation's target name.

    Returns
    -------
    planet_letters: List of strings
        List of all planets (known to be) targetted by an observation.

    Examples
    --------
    >>> import gen_tso.catalogs as cat
    >>> obs = cat.parse_program(pid=3712)

    >>> targets = [
    >>>     target for target in cat.Catalog().targets
    >>>     if target.is_transiting
    >>> ]
    >>> letters = cat.get_planet_letters(obs[0], targets)
    """
    pid = obs["pid"]
    obs_id = obs["observation"]
    visit = obs["visit"]
    info = f'{pid} {obs_id:3} {visit}  '

    # Start with the exceptions, hardcoded patching:
    if pid=='5191':
        return ['b', 'c']
    if pid=='5177':
        return ['b']
    if pid=='8739':
        return ['b']
    if pid=='2420' and obs_id=='5' and visit=='1':
        return ['c']
    if pid=='3818' and obs_id=='2' and visit=='1':
        return ['d']
    if pid=='9235' and obs_id=='5':
        return ['b']

    target_name = obs['target']
    # The planet is in the 'target'
    if target_name[-1].lower() == 'b' and not target_name[-2].isalpha():
        name = target_name[:-1]
        planet_letters = ['b']
        if verbose:
            print(f'{info}{target_name:15}  {planet_letters}')
        return planet_letters

    # Cross-check target with catalog:
    name = normalize_name(target_name)
    # Find host / planets
    planets = []
    for target in targets:
        aliases = [target.planet] + target.aliases
        hosts = [get_host(alias) for alias in aliases]
        if name in hosts:
            planets.append(target)

    # Single-planet systems
    if len(planets) == 1:
        planet_letters = [get_letter(planets[0].planet).strip()]
        return planet_letters

    # From label
    if len(planets) > 0:
        obs_label = obs['label']
        hosts = [planets[0].host] + [get_host(a) for a in planets[0].aliases]
        label = _clean_label(obs_label, hosts)
        planet_letters = _planet_from_label(label)
        if len(planet_letters) > 0:
            if verbose:
                print(f'{info}{repr(obs_label)}  {planet_letters}')
            return planet_letters

        # From period
        if obs['period'] is not None:
            planet_letters = [_planet_from_period(obs, planets)]
            if verbose:
                period = obs["period"]
                periods = np.round([p.period for p in planets], 1)
                print(f'{info}{name:15}{period:6.1f}  {periods}  {planet_letters}')
            return planet_letters

    print(f'Could not determine planet for {repr(name)} (PID={pid}, {obs_id})')
    return []

