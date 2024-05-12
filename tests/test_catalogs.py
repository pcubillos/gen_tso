import gen_tso.catalogs as cat
import numpy as np


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

