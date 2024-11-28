import pytest
import gen_tso.catalogs as cat
import numpy as np
from gen_tso.utils import ROOT


def test_fetch_nea_aliases_single():
    targets = 'KELT-7'
    host_aliases, planet_aliases = cat.fetch_nea_aliases(targets)
    # TBD: assert
    assert planet_aliases == [{
        'HD 33643 b': 'KELT-7 b',
        'KELT-7 b': 'KELT-7 b',
        'BD+33 977 b': 'KELT-7 b',
        'HIP 24323 b': 'KELT-7 b',
        'SAO 57753 b': 'KELT-7 b',
        'GSC 2393-00852 b': 'KELT-7 b',
        'TYC 2393-00852-1 b': 'KELT-7 b',
        '2MASS J05131092+3319054 b': 'KELT-7 b',
        'WISE J051310.93+331904.8 b': 'KELT-7 b',
        'TIC 367366318 b': 'KELT-7 b',
        'Gaia DR2 181908842994567936 b': 'KELT-7 b',
        'TOI-1682.01': 'KELT-7 b',
        'TOI-1682 b': 'KELT-7 b',
    }]
    assert host_aliases == [{
        'TIC 367366318': 'KELT-7',
        'Gaia DR2 181908842994567936': 'KELT-7',
        'TOI-1682': 'KELT-7',
        'HD 33643': 'KELT-7',
        'KELT-7': 'KELT-7',
        'BD+33 977': 'KELT-7',
        'HIP 24323': 'KELT-7',
        'SAO 57753': 'KELT-7',
        'GSC 2393-00852': 'KELT-7',
        'TYC 2393-00852-1': 'KELT-7',
        '2MASS J05131092+3319054': 'KELT-7',
        'WISE J051310.93+331904.8': 'KELT-7',
    }]


def test_fetch_nea_aliases_list():
    targets = ['WASP-8 b', 'KELT-7', 'HD 189733']
    host_aliases, planet_aliases = cat.fetch_nea_aliases(targets)
    # TBD: assert


def test_fetch_nea_aliases_fail():
    targets = 'WASP-999'
    host_aliases, planet_aliases = cat.fetch_nea_aliases(targets)
    assert host_aliases == [{}]
    assert planet_aliases == [{}]


def test_fetch_aliases_oddballs():
    # import grequests
    # import requests
    # import urllib
    # from gen_tso.utils import ROOT
    # import gen_tso.catalogs.utils as u
    # from gen_tso.catalogs import *
    hosts = [
        'TOI-4336',
        'TOI-216',
        'WASP-50',
        'WASP-53',
        'WASP-76',
        '55 Cnc',
    ]
    host_aliases, planet_aliases = fetch_nea_aliases(hosts)


def test_fetch_gaia_targets():
    target_ra = 315.0260
    target_dec = -5.0949
    names, G_mag, teff, log_g, ra, dec, separation = cat.fetch_gaia_targets(
        target_ra, target_dec, max_separation=80.0,
    )
    names = np.array([
        'Gaia DR3 6910753016653587840', 'Gaia DR3 6910752844854895360',
        'Gaia DR3 6910747136843460480', 'Gaia DR3 6910746934979897088',
        'Gaia DR3 6910747141138328064', 'Gaia DR3 6910753046718453248',
        'Gaia DR3 6910746930684973952', 'Gaia DR3 6910746866260418944',
        'Gaia DR3 6910746934979895936',
    ])
    G_mag = np.array([
         9.491504, 16.80548 , 18.476078, 16.334204, 18.974197, 17.175997,
        18.024723, 15.700529, 16.899416,
    ])
    teff = np.array([
        4729.3774, 5503.5996, 4959.2495, 5394.8145, 3979.5588, 5134.7896,
        4528.2563, 5572.225 , 4862.2915,
    ])
    log_g = np.array([
        4.5058, 4.4191, 4.7718, 4.3786, 5.018 , 4.3886, 4.6305, 3.774 ,
        4.6665,
    ])
    ra = np.array([
        315.02597081, 315.03419028, 315.01496413, 315.01522348,
        315.00966205, 315.04041427, 315.02272526, 315.03525381,
        315.0218094,
    ])
    dec = np.array([
        -5.09487006, -5.08816924, -5.09282223, -5.10531022, -5.10264537,
        -5.08275871, -5.11378437, -5.11325095, -5.11603216,
    ])
    sep = np.array([
        4.99691642e-02, 3.80699861e+01, 4.01249572e+01, 5.38511968e+01,
        6.48380986e+01, 6.76834547e+01, 6.91223563e+01, 7.41212133e+01,
        7.76740049e+01,
    ])


@pytest.mark.skip(reason='mock requests')
def test_fetch_gaia_targets_error():
    pass

