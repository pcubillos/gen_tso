import pathlib
import numpy as np
from gen_tso.utils import ROOT
import gen_tso.catalogs.utils as u

# Path to tests folder
ROOT = pathlib.Path(ROOT).parent.joinpath('tests')


def test_get_trexolists_targets_default():
    targets = u.get_trexolists_targets(
        trexo_file=f'{ROOT}/mocks/trexolists.csv',
    )
    expected_targets = [
        'CD-38 2551', 'GJ 341', 'TRAPPIST-1', 'WASP-107', 'WASP-63',
    ]
    np.testing.assert_equal(targets, expected_targets)


def test_get_trexolists_targets_group():
    targets = u.get_trexolists_targets(
        trexo_file=f'{ROOT}/mocks/trexolists.csv',
        grouped=True,
    )
    expected_targets = [
        ['GJ 341'],
        ['WASP-107'],
        ['TRAPPIST-1'],
        ['CD-38 2551', 'WASP-63'],
    ]
    np.testing.assert_equal(targets, expected_targets)


def test_get_trexolists_targets_coords():
    targets, ra, dec = u.get_trexolists_targets(
        trexo_file=f'{ROOT}/mocks/trexolists.csv',
        extract='coords',
    )
    expected_targets = [
        'CD-38 2551', 'GJ 341', 'TRAPPIST-1', 'WASP-107', 'WASP-63',
    ]
    expected_ra = ['06:17:2', '09:21:3', '23:06:3', '12:33:3', '06:17:2']
    expected_dec = ['-38:19', '-60:16', '-05:02', '-10:08', '-38:19']
    np.testing.assert_equal(targets, expected_targets)
    np.testing.assert_equal(ra, expected_ra)
    np.testing.assert_equal(dec, expected_dec)


def test_get_trexolists_targets_group_coords():
    targets, ra, dec = u.get_trexolists_targets(
        trexo_file=f'{ROOT}/mocks/trexolists.csv',
        grouped=True,
        extract='coords',
    )
    expected_targets = [
        ['GJ 341'],
        ['WASP-107'],
        ['TRAPPIST-1'],
        ['CD-38 2551', 'WASP-63'],
    ]
    expected_ra = ['09:21:3', '12:33:3', '23:06:3', '06:17:2']
    expected_dec = ['-60:16', '-10:08', '-05:02', '-38:19']
    np.testing.assert_equal(targets, expected_targets)
    np.testing.assert_equal(ra, expected_ra)
    np.testing.assert_equal(dec, expected_dec)

