import gen_tso.catalogs as cat
import numpy as np
from gen_tso.utils import ROOT


def test_fetch_nea_aliases_single():
    targets = 'KELT-7'
    host_aliases, planet_aliases = cat.fetch_nea_aliases(targets)
    # TBD: assert


def test_fetch_nea_aliases_list():
    targets = ['WASP-8 b', 'KELT-7', 'HD 189733']
    host_aliases, planet_aliases = cat.fetch_nea_aliases(targets)
    # TBD: assert


def test_fetch_nea_aliases_fail():
    targets = 'WASP-999'
    host_aliases, planet_aliases = cat.fetch_nea_aliases(targets)
    # TBD: assert

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


