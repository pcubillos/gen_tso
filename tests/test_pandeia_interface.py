# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import gen_tso.pandeia_io as jwst


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



