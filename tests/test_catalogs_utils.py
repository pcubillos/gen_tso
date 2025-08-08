# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import pathlib
import numpy as np
from gen_tso.utils import ROOT
import gen_tso.catalogs.utils as u

# Path to tests folder
ROOT = pathlib.Path(ROOT).parent.joinpath('tests')


def test_esasky_js_circle():
    ra = 315.0259661
    dec = -5.094857
    radius = 80.0
    circle = u.esasky_js_circle(ra, dec, radius)


    expected_circle = {
        'event': 'overlayFootprints',
        'content': {
            'overlaySet': {
                'type': 'FootprintListOverlay',
                'overlayName': 'visit splitting distance',
                'cooframe': 'J2000',
                'color': '#15B01A',
                'lineWidth': 5,
                'skyObjectList': [{
                     'name': 'visit splitting distance',
                     'id': 1,
                     'stcs': 'CIRCLE ICRS 315.02596610 -5.09485700 0.0222',
                     'ra_deg': '315.02596610',
                     'dec_deg': '-5.09485700',
                }]
            }
        }
    }
    assert circle == expected_circle


def test_json_target_property():
    name = 'G mag'
    value = 9.491504
    format = '.2f'
    prop = u.json_target_property(name, value, format)
    expected_prop = {'name': 'G mag', 'value': '9.49', 'type': 'STRING'}
    assert prop == expected_prop


def test_json_target():
    # e.g., from a cat.fetch_gaia_targets(ra_source, dec_source) call
    i = 0
    name = 'Gaia DR3 6910753016653587840'
    g_mag = 9.491504
    teff = 4729.3774
    logg = 4.5058
    ra = 315.02597081
    dec = -5.09487006
    separation = 0.15023734
    j_target = u.json_target(i, name, ra, dec, g_mag, teff, logg, separation)

    expected_target = {
        'name': 'Gaia DR3 6910753016653587840',
        'id': 1,
        'ra': '315.02597081',
        'dec': '-5.09487006',
        'data': [
            {'name': 'G mag', 'value': '9.49', 'type': 'STRING'},
            {'name': 'T eff', 'value': '4729.4', 'type': 'STRING'},
            {'name': 'log(g)', 'value': '4.51', 'type': 'STRING'},
            {'name': 'Separation', 'value': '0.150', 'type': 'STRING'}
        ]
    }
    assert j_target == expected_target


def test_esasky_js_catalog():
    pass


def test_normalize_name_odd():
    assert u.normalize_name('-RHO01-CNC') == '55 Cnc'
    assert u.normalize_name('-rho01-Cnc') == '55 Cnc'
    assert u.normalize_name('55CNC') == '55 Cnc'

    assert u.normalize_name('NAME-G-268-38B') == 'G 268-38'

    assert u.normalize_name('WD-1856+534-B') == 'WD 1856+534'
    assert u.normalize_name('WD1856+534') == 'WD 1856+534'
    assert u.normalize_name('WD1856B') == 'WD 1856+534'

    assert u.normalize_name('V-V1298-TAU') == 'V1298 Tau'
    assert u.normalize_name('V-V1298-Tau') == 'V1298 Tau'


def test_normalize_name_tails():
    assert u.normalize_name('CD-38-1467-copy') == 'CD-38 1467'
    assert u.normalize_name('GJ-436-offset') ==  'GJ 436'
    assert u.normalize_name('L98-59-updated') == 'L 98-59'
    assert u.normalize_name('TOI-455-revised') == 'TOI-455'


def test_normalize_name_case():
    assert u.normalize_name('KEPLER-12') == 'Kepler-12'
    assert u.normalize_name('KEPLER-86B') == 'Kepler-86'
    assert u.normalize_name('Kepler-12') == 'Kepler-12'
    assert u.normalize_name('TRES-4') == 'TrES-4'
    assert u.normalize_name('WOLF-437') == 'Wolf 437'


def test_normalize_name_BD():
    assert u.normalize_name('BD+01-316') == 'BD+01 316'
    assert u.normalize_name('BD-17-588A') == 'BD-17 588 A'


def test_normalize_name_HATs():
    assert u.normalize_name('HAT-P-1') == 'HAT-P-1'
    assert u.normalize_name('HAT-P-26B') == 'HAT-P-26'
    assert u.normalize_name('HATP1') == 'HAT-P-1'
    assert u.normalize_name('HATS-6') == 'HATS-6'
    assert u.normalize_name('HATS-72b') == 'HATS-72'


def test_normalize_name_HDs():
    assert u.normalize_name('HD-12572b') == 'HD 12572'
    assert u.normalize_name('HD-133112') == 'HD 133112'
    assert u.normalize_name('HD-189733B') == 'HD 189733'
    assert u.normalize_name('HD106315') == 'HD 106315'


def test_normalize_name_GJs():
    assert u.normalize_name('GJ-1132') == 'GJ 1132'
    assert u.normalize_name('GJ-4102A') == 'GJ 4102 A'
    assert u.normalize_name('GJ1132') == 'GJ 1132'
    assert u.normalize_name('GL486') == 'GJ 486'


def test_normalize_name_Ls():
    assert u.normalize_name('L-231-32') == 'L 231-32'
    assert u.normalize_name('L168-9') == 'L 168-9'

    assert u.normalize_name('LHS-1140') == 'LHS 1140'
    assert u.normalize_name('LHS1140') == 'LHS 1140'
    assert u.normalize_name('LHS_3844') == 'LHS 3844'

    assert u.normalize_name('LP-141-14') == 'LP 141-14'

    assert u.normalize_name('LTT-1445A') == 'LTT 1445 A'
    assert u.normalize_name('LTT-3780') == 'LTT 3780'
    assert u.normalize_name('LTT1445A') == 'LTT 1445 A'
    assert u.normalize_name('LTT9779') == 'LTT 9779'


def test_normalize_name_TOIs():
    assert u.normalize_name('TOI-1075') == 'TOI-1075'
    assert u.normalize_name('TOI-1807b') == 'TOI-1807'


def test_normalize_name_K2s():
    assert u.normalize_name('K2-141') == 'K2-141'
    assert u.normalize_name('K2-22B') == 'K2-22'


def test_normalize_name_WASPs():
    assert u.normalize_name('WASP-103') == 'WASP-103'
    assert u.normalize_name('WASP-121B') == 'WASP-121'
    assert u.normalize_name('WASP-39b') == 'WASP-39'
    assert u.normalize_name('WASP-77A') == 'WASP-77 A'


def test_normalize_name_others():
    assert u.normalize_name('2MASS-J11335277+1227034') == '2MASS J11335277+1227034'
    assert u.normalize_name('AU-MIC') == 'AU Mic'
    assert u.normalize_name('CD-38-1467') == 'CD-38 1467'
    assert u.normalize_name('HIP67522') == 'HIP 67522'
    assert u.normalize_name('IRAS-04125+2902') == 'IRAS 04125+2902'
    assert u.normalize_name('KELT-20') == 'KELT-20'
    assert u.normalize_name('NGTS-10') == 'NGTS-10'
    assert u.normalize_name('TRAPPIST-1') == 'TRAPPIST-1'
    assert u.normalize_name('TRAPPIST-1B') == 'TRAPPIST-1'
    assert u.normalize_name('TYC-7052-1753-1') == 'TYC 7052-1753-1'


def test_is_letter_yes():
    assert u.is_letter('WASP-69 b') is True


def test_is_letter_no():
    assert u.is_letter('TOI-741.01') is False
    assert u.is_letter('WASP-69') is False
    # TBD: check
    assert u.is_letter('WASP-8 A') is False


def test_is_candidate_yes():
    assert u.is_candidate('TOI-741.01') is True


def test_is_candidate_no():
    assert u.is_candidate('WASP-69 b') is False
    assert u.is_candidate('WASP-69') is False
    assert u.is_candidate('WASP-8 A') is False


def test_get_letter_ok():
    assert u.get_letter('WASP-69 b') == ' b'
    assert u.get_letter('TOI-741.01') == '.01'

def test_get_letter_fail():
    assert u.get_letter('WASP-69') == ''
    assert u.get_letter('WASP-8 A') == ''

def test_get_host_ok():
    assert u.get_host('WASP-69 b') == 'WASP-69'
    assert u.get_host('TOI-741.01') == 'TOI-741'


def test_get_host_fail():
    assert u.get_host('WASP-69') == ''
    # TBD: is this a problem?
    assert u.get_host('WASP-8 A') == ''


def test_to_float_float_int():
    value = u.to_float('101')
    np.testing.assert_equal(value, 101.0)


def test_to_float_float():
    value = u.to_float('1.01')
    np.testing.assert_equal(value, 1.01)


def test_to_float_exp():
    value = u.to_float('3.0e3')
    np.testing.assert_equal(value, 3000.0)


def test_to_float_none():
    assert u.to_float('None') is None


def as_str_ok():
    string = u.as_str(1.0)
    assert string == '1.0'


def as_str_exp():
    string = u.as_str(3.0e2)
    assert string == '300.0'


def as_str_nan():
    string = u.as_str(np.nan)
    assert string is None


def as_str_my_nan():
    string = u.as_str(np.nan, if_none='---')
    assert string == '---'


def as_str_fmt():
    string = u.as_str(1.0, fmt='.3f')
    assert string == '1.000'


