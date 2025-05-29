# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import pytest
import numpy as np
from gen_tso.catalogs.fetch_catalogs import (
    fetch_simbad_aliases,
    fetch_nea_aliases,
    fetch_aliases,
)


@pytest.mark.parametrize(
    'target',
    [
        'WASP-69',
        ['WASP-69'],
    ],
)
def test_fetch_simbad_aliases_single(target):
    aliases, ks_mag = fetch_simbad_aliases(target)

    expected_aliases = [
        'AP J21000618-0505398',
        'NAME Wouri',
        'Gaia DR3 6910753016653587840',
        'TIC 248853232',
        'BD-05  5432',
        'GSC 05200-01560',
        'PPM 204700',
        'TYC 5200-1560-1',
        'Gaia DR2 6910753016653587840',
        '2MASS J21000618-0505398',
        'WASP-69',
        'Gaia DR1 6910753012357203968',
    ]
    assert len(aliases[0]) == len(expected_aliases)
    assert aliases[0] == expected_aliases
    np.testing.assert_allclose(ks_mag, 7.45900011)


def test_fetch_simbad_aliases_planets():
    # Simbad thinks the TrES targets are the planets
    targets = ['TrES-4']
    aliases, ks_mag = fetch_simbad_aliases(targets)
    expected_aliases = [
        'TOI-2124',
        'TOI-2124',
        'TrES-4',
        'HIDDEN NAME TrES-4',
    ]
    assert len(aliases[0]) == len(expected_aliases)
    assert aliases[0] == expected_aliases
    assert np.isnan(ks_mag[0])


def test_fetch_simbad_aliases_not_found():
    aliases, ks_mag = fetch_simbad_aliases(['WASP-00'])
    assert aliases[0] == []
    assert np.isnan(ks_mag[0])


def test_fetch_nea_aliases_basics():
    targets = ['WASP-8 b', 'KELT-7']
    host_aliases, planet_aliases = fetch_nea_aliases(targets)

    assert len(host_aliases) == 2
    assert len(planet_aliases) == 2

    expected_host_aliases = {
        '1SWASP J235936.07-350152.9',
        '2MASS J23593607-3501530',
        'Gaia DR2 2312679845530628096',
        'TIC 183532609',
        'TOI-191',
        'TYC 7522-00505-1',
        'WASP-8',
        'WASP-8 A',
        'WISE J235936.16-350153.1',
    }
    assert set(host_aliases[0]) == expected_host_aliases 
    assert np.unique(list(host_aliases[0].values())) == 'WASP-8'
    assert np.unique(list(host_aliases[1].values())) == 'KELT-7'

    expected_planets_wasp = set(planet_aliases[0].values())
    expected_planets_kelt = set(planet_aliases[1].values())
    assert expected_planets_wasp == {'WASP-8 b', 'WASP-8 c'}
    assert expected_planets_kelt == {'KELT-7 b'}


def test_fetch_nea_aliases_multistar_systems():
    targets = [
        'WASP-8',   # 2x star, only one has two planets 
        'TOI-1338', # 2x star, 2 planets transiting both stars
        'WASP-94',  # 2x star, 1 planet each
    ]
    host_aliases, planet_aliases = fetch_nea_aliases(targets)

    assert len(host_aliases) == 3
    assert len(planet_aliases) == 3

    # Check identifiers
    expected_hosts = [
        set(['WASP-8']),
        set(['TOI-1338 A', 'TOI-1338 B']),
        set(['WASP-94 A', 'WASP-94 B']),
    ]
    assert set(host_aliases[0].values()) == expected_hosts[0]
    assert set(host_aliases[1].values()) == expected_hosts[1]
    assert set(host_aliases[2].values()) == expected_hosts[2]

    expected_planets = [
        set(['WASP-8 b', 'WASP-8 c']),
        set(['TOI-1338 b', 'TOI-1338 c']),
        set(['WASP-94 A b', 'WASP-94 B b']),
    ]
    assert set(planet_aliases[0].values()) == expected_planets[0]
    assert set(planet_aliases[1].values()) == expected_planets[1]
    assert set(planet_aliases[2].values()) == expected_planets[2]

    # Check planet aliases WASP-8:
    expected_aliases_WASP8b = set([
        'WASP-8 b',
        'TYC 7522-00505-1 b',
        '2MASS J23593607-3501530 b',
        'WISE J235936.16-350153.1  b',
        '1SWASP J235936.07-350152.9 b',
        'TIC 183532609 b',
        'Gaia DR2 2312679845530628096 b',
        'TOI-191.01',
        'TOI-191 b',
    ])
    expected_aliases_WASP8c = set([
        'TIC 183532609 c',
        'Gaia DR2 2312679845530628096 c',
        'WASP-8 c',
        'WASP-8 A b',
        'WASP-8 A c',
        'TYC 7522-00505-1 c',
        '2MASS J23593607-3501530 c',
        'WISE J235936.16-350153.1  c',
        '1SWASP J235936.07-350152.9 c',
    ])
    aliases_WASP8b = set([
        key for key,val in planet_aliases[0].items()
        if val == 'WASP-8 b'
    ])
    aliases_WASP8c = set([
        key for key,val in planet_aliases[0].items()
        if val == 'WASP-8 c'
    ])
    assert aliases_WASP8b == expected_aliases_WASP8b
    assert aliases_WASP8c == expected_aliases_WASP8c

    # Check planet aliases WASP-94:
    expected_aliases_WASP94Ab = set([
        'WASP-94 A b',
        'TYC 7466-1400-1 b',
        '2MASS J20550794-3408079 b',
        'WISE J205507.96-340808.4 b',
        '1SWASP J205507.94-340807.9 A b',
        'TIC 92352620 b',
        'Gaia DR2 6780546169633475456 b',
        'TOI-107.01',
        'TOI-107 b',
    ])
    expected_aliases_WASP94Bb = set([
        'WASP-94 B b',
        '2MASS J20550915-3408078 b',
        'WISE J205509.17-340808.3 b',
        '1SWASP J205507.94-340807.9 B b',
        'TIC 92352621 b',
        'Gaia DR2 6780546169633474944 b',
    ])
    aliases_WASP94Ab = set([
        key for key,val in planet_aliases[2].items()
        if val == 'WASP-94 A b'
    ])
    aliases_WASP94Bb = set([
        key for key,val in planet_aliases[2].items()
        if val == 'WASP-94 B b'
    ])
    assert aliases_WASP94Ab == expected_aliases_WASP94Ab
    assert aliases_WASP94Bb == expected_aliases_WASP94Bb


def test_fetch_nea_aliases_not_found():
    host_aliases, planet_aliases = fetch_nea_aliases('WASP-00')
    assert host_aliases[0] == {}
    assert planet_aliases[0] == {}


def test_fetch_aliases_basics():
    hosts = [
        'WASP-69',
        'TOI-270',
    ]
    aliases = fetch_aliases(hosts)

    assert list(aliases) == hosts

    expected_keys = ['host', 'planets', 'host_aliases', 'planet_aliases']
    for key in expected_keys:
        assert key in aliases[hosts[0]].keys()

    expected_toi_planets = ['TOI-270 b', 'TOI-270 c', 'TOI-270 d']
    assert aliases[hosts[0]]['planets'] == ['WASP-69 b']
    assert list(aliases[hosts[1]]['planets']) == expected_toi_planets

    expected_planet_aliases = {
        'WASP-69 b': 'WASP-69 b',
        'BD-05 5432 b': 'WASP-69 b',
        'GSC 05200-01560 b': 'WASP-69 b',
        'TYC 5200-1560-1 b': 'WASP-69 b',
        '2MASS J21000618-0505398 b': 'WASP-69 b',
        'WISE J210006.21-050540.9 b': 'WASP-69 b',
        'TIC 248853232 b': 'WASP-69 b',
        'Gaia DR2 6910753016653587840 b': 'WASP-69 b',
        'TOI-5823.01': 'WASP-69 b',
        'TOI-5823 b': 'WASP-69 b',
        'Wouri b': 'WASP-69 b',
    }
    assert aliases[hosts[0]]['planet_aliases'] == expected_planet_aliases


def test_fetch_aliases_multiplanet_systems():
    hosts = ['WASP-8', 'TOI-1201', 'TOI-1338 A']
    aliases = fetch_aliases(hosts)

    assert hosts[0] in aliases
    assert hosts[1] in aliases
    assert hosts[2] in aliases

    assert set(aliases[hosts[0]]['planets']) == {'WASP-8 b', 'WASP-8 c'}
    assert set(aliases[hosts[1]]['planets']) == {'TOI-1201 b'}
    assert set(aliases[hosts[2]]['planets']) == {'TOI-1338 b', 'TOI-1338 c'}

    host_aliases_WASP8 = {
        '1SWASP J235936.07-350152.9',
        '2MASS J23593607-3501530',
        'Gaia DR2 2312679845530628096',
        'TIC 183532609',
        'TOI-191',
        'TYC 7522-00505-1',
        'WASP-8',
        'WASP-8 A',
        'WISE J235936.16-350153.1',
    }
    host_aliases_TOI1201 = {
        '2MASS J02485926-1432152',
        'Gaia DR2 5157183324996790272',
        'TIC 29960110',
        'TOI-1201',
    }
    host_aliases_TOI1338 = {
        '2MASS J06083197-5932280',
        'BEBOP-1',
        'EBLM J0608-59',
        'Gaia DR2 5494443978353833088',
        'TIC 260128333',
        'TOI-1338 A',
        'TOI-1338 Aa',
        'TYC 8533-950-1',
        'WISE J060831.94-593227.6',
    }
    assert set(aliases[hosts[0]]['host_aliases']) == host_aliases_WASP8
    assert set(aliases[hosts[1]]['host_aliases']) == host_aliases_TOI1201
    assert set(aliases[hosts[2]]['host_aliases']) == host_aliases_TOI1338

    planet_aliases_WASP8 = {
        'WASP-8 b': 'WASP-8 b',
        'TYC 7522-00505-1 b': 'WASP-8 b',
        '2MASS J23593607-3501530 b': 'WASP-8 b',
        'WISE J235936.16-350153.1 b': 'WASP-8 b',
        '1SWASP J235936.07-350152.9 b': 'WASP-8 b',
        'TIC 183532609 b': 'WASP-8 b',
        'Gaia DR2 2312679845530628096 b': 'WASP-8 b',
        'TOI-191.01': 'WASP-8 b',
        'TOI-191 b': 'WASP-8 b',
        'TIC 183532609 c': 'WASP-8 c',
        'Gaia DR2 2312679845530628096 c': 'WASP-8 c',
        'WASP-8 c': 'WASP-8 c',
        'WASP-8 A b': 'WASP-8 b',
        'WASP-8 A c': 'WASP-8 c',
        'TYC 7522-00505-1 c': 'WASP-8 c',
        '2MASS J23593607-3501530 c': 'WASP-8 c',
        'WISE J235936.16-350153.1 c': 'WASP-8 c',
        '1SWASP J235936.07-350152.9 c': 'WASP-8 c',
    }
    planet_aliases_TOI1201 = {
        'TOI-1201 b': 'TOI-1201 b',
        'TOI-1201.01': 'TOI-1201 b',
        'TIC 29960110 b': 'TOI-1201 b',
        'TIC 29960110.01': 'TOI-1201 b',
        '2MASS J02485926-1432152 b': 'TOI-1201 b',
        'Gaia DR2 5157183324996790272 b': 'TOI-1201 b',
    }
    planet_aliases_TOI1338 = {
        'Gaia DR2 5494443978353833088 b': 'TOI-1338 b',
        'TOI-1338 b': 'TOI-1338 b',
        'TOI-1338.01': 'TOI-1338 b',
        'EBLM J0608-59 b': 'TOI-1338 b',
        'TIC 260128333 b': 'TOI-1338 b',
        'TIC 260128333.01': 'TOI-1338 b',
        'TYC 8533-950-1 b': 'TOI-1338 b',
        '2MASS J06083197-5932280 b': 'TOI-1338 b',
        'WISE J060831.94-593227.6 b': 'TOI-1338 b',
        'TOI-1338 c': 'TOI-1338 c',
        'TIC 260128333 c': 'TOI-1338 c',
        '2MASS J06083197-5932280 c': 'TOI-1338 c',
        'WISE J060831.94-593227.6 c': 'TOI-1338 c',
        'Gaia DR2 5494443978353833088 c': 'TOI-1338 c',
        'TYC 8533-950-1 c': 'TOI-1338 c',
        'EBLM J0608-59 c': 'TOI-1338 c',
        'BEBOP-1 c': 'TOI-1338 c',
        'BEBOP-1 b': 'TOI-1338 b',
    }
    assert aliases[hosts[0]]['planet_aliases'] == planet_aliases_WASP8
    assert aliases[hosts[1]]['planet_aliases'] == planet_aliases_TOI1201
    assert aliases[hosts[2]]['planet_aliases'] == planet_aliases_TOI1338


def test_fetch_aliases_name_change():
    hosts = [
        'TOI-1241',  # KOI-5
        'TOI-2410',  # EPIC 220198551
        'TOI-4454',  # Kepler-488
        'TOI-4588',  # HIP 92247
        'TOI-7383',  # Gaia-1
    ]
    aliases = fetch_aliases(hosts)

    expected_hosts = [
        'KOI-5',
        'EPIC 220198551',
        'Kepler-488',
        'HIP 92247',
        'Gaia-1',
    ]
    assert list(aliases) == expected_hosts
    for i in range(len(hosts)):
        host = expected_hosts[i]
        host_aliases = aliases[host]['host_aliases']
        assert hosts[i] in host_aliases
