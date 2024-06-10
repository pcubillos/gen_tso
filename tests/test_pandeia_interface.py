# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import os
import pickle

import numpy as np

import gen_tso.pandeia_io as jwst
from gen_tso.utils import ROOT

os.chdir(ROOT+'../tests')


def test_saturation_level_perform_calculation_single():
    # See tests/make_mocks.py
    with open('mocks/perform_calculation_nircam_ssgrism.pkl', 'rb') as f:
        result = pickle.load(f)

    pixel_rate, full_well = jwst.saturation_level(result)
    expected_rate = 1396.5528564453125
    expected_well = 58100.00422843957
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_saturation_level_perform_calculation_multi():
    # See tests/make_mocks.py
    with open('mocks/perform_calculation_miri_mrs_ts.pkl', 'rb') as f:
        result = pickle.load(f)

    pixel_rate, full_well = jwst.saturation_level(result)
    expected_rate = [
        204.37779236, 109.17165375,  32.75593948,   4.55676889,
    ]
    expected_well = [
        193654.99564498, 193655.00540975, 193655.004117  , 193654.99798687,
    ]
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_saturation_level_tso_calculation_single():
    # See tests/make_mocks.py
    with open('mocks/tso_calculation_nircam_ssgrism.pkl', 'rb') as f:
        tso = pickle.load(f)

    pixel_rate, full_well = jwst.saturation_level(tso)
    expected_rate = [1354.6842041 , 1396.55285645]
    expected_well = [58100.00016363, 58100.00129109]
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_saturation_level_tso_calculation_multi():
    # See tests/make_mocks.py
    with open('mocks/tso_calculation_miri_mrs_ts.pkl', 'rb') as f:
        tso = pickle.load(f)

    pixel_rate, full_well = jwst.saturation_level(tso)
    expected_rate = [
        198.24740601, 105.90003204,  31.80119514,   4.46783495,
        204.37779236, 109.17165375,  32.75593948,   4.55676889,
    ]
    expected_well = [
        193654.99147199, 193654.99886718, 193654.98493823, 193654.99820043,
        193654.99564498, 193655.00540975, 193655.004117  , 193654.99798687,
    ]
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_saturation_level_tso_calculation_get_max():
    # TBD: implement
    pass


def test_exposure_time_nircam():
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'rapid'
    ngroup = 90
    nint = 1
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    np.testing.assert_allclose(exp_time, 31.00063)


def test_exposure_time_miri():
    inst = 'miri'
    subarray = 'slitlessprism'
    readout = 'fastr1'
    ngroup = 30
    nint = 1
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    np.testing.assert_allclose(exp_time, 4.7712)


def test_exposure_time_bad_subarray():
    inst = 'nircam'
    subarray = 'nope'
    readout = 'rapid'
    ngroup = 90
    nint = 1
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    assert exp_time == 0.0


def test_exposure_time_bad_readout():
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'nope'
    ngroup = 90
    nint = 1
    exp_time = jwst.exposure_time(inst, subarray, readout, ngroup, nint)
    assert exp_time == 0.0


def test_bin_search_exposure_time_nircam():
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'rapid'
    ngroup = 90
    obs_time = 6.0
    nint, exp_time = jwst.bin_search_exposure_time(
        inst, subarray, readout, ngroup, obs_time,
    )
    assert nint == 697
    np.testing.assert_allclose(exp_time, 21607.43911)


def test_bin_search_exposure_time_miri():
    inst = 'miri'
    subarray = 'slitlessprism'
    readout = 'fastr1'
    ngroup = 30
    obs_time = 6.0
    nint, exp_time = jwst.bin_search_exposure_time(
        inst, subarray, readout, ngroup, obs_time,
    )
    assert nint == 4382
    np.testing.assert_allclose(exp_time, 21604.15264)


def test_bin_search_exposure_time_bad_subarray():
    inst = 'nircam'
    subarray = 'nope'
    readout = 'rapid'
    ngroup = 90
    obs_time = 6.0
    nint, exp_time = jwst.bin_search_exposure_time(
        inst, subarray, readout, ngroup, obs_time,
    )
    assert exp_time == 0.0


def test_bin_search_exposure_time_bad_readout():
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'nope'
    ngroup = 90
    obs_time = 6.0
    nint, exp_time = jwst.bin_search_exposure_time(
        inst, subarray, readout, ngroup, obs_time,
    )
    assert exp_time == 0.0


def test__print_pandeia_saturation_perform_calc():
    # See tests/make_mocks.py
    with open('mocks/perform_calculation_nircam_ssgrism.pkl', 'rb') as f:
        result = pickle.load(f)

    text = jwst._print_pandeia_saturation(reports=[result], format=None)
    expected_text = """Max fraction of saturation: 1.6%
ngroup below 80% saturation: 97
ngroup below 100% saturation: 122"""
    assert text == expected_text


def test__print_pandeia_saturation_tso_calc():
    # See tests/make_mocks.py
    with open('mocks/tso_calculation_miri_mrs_ts.pkl', 'rb') as f:
        tso = pickle.load(f)

    text = jwst._print_pandeia_saturation(reports=tso, format=None)
    expected_text = """Max fraction of saturation: 73.2%
ngroup below 80% saturation: 273
ngroup below 100% saturation: 341"""
    assert text == expected_text


def test__print_pandeia_saturation_values():
    pixel_rate, full_well = 1396.5528564453125, 58100.002280515175
    inst = 'nircam'
    subarray = 'subgrism64'
    readout = 'rapid'
    ngroup = 90
    text = jwst._print_pandeia_saturation(
        inst, subarray, readout, ngroup, pixel_rate, full_well,
        format=None,
    )
    expected_text = """Max fraction of saturation: 73.7%
ngroup below 80% saturation: 97
ngroup below 100% saturation: 122"""
    assert text == expected_text

