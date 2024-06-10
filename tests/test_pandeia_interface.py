# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import gen_tso.pandeia_io as jwst


def test_saturation_level_perform_calculation_single():
    # TBD: mock pando.perform_calculation()
    pando = jwst.PandeiaCalculation('nircam', 'ssgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    result = pando.perform_calculation(
        ngroup=2, nint=683, readout='rapid', filter='f444w',
    )

    pixel_rate, full_well = jwst.saturation_level(result)
    expected_rate = 1396.5528564453125
    expected_well = 58100.00422843957
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_saturation_level_perform_calculation_multi():
    # TBD: mock pando.perform_calculation()
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    aperture = ['ch1', 'ch2', 'ch3', 'ch4']
    result = pando.perform_calculation(
        ngroup=250, nint=40, aperture=aperture,
    )

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
    # TBD: mock pando.tso_calculation()
    wl = np.logspace(0, 2, 1000)
    depth = [wl, np.tile(0.03, len(wl))]
    pando = jwst.PandeiaCalculation('nircam', 'ssgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    tso = pando.tso_calculation(
        'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
        ngroup=90, readout='rapid', filter='f444w',
    )

    pixel_rate, full_well = jwst.saturation_level(tso)
    expected_rate = [1354.6842041 , 1396.55285645]
    expected_well = [58100.00016363, 58100.00129109]
    np.testing.assert_allclose(pixel_rate, expected_rate)
    np.testing.assert_allclose(full_well, expected_well)


def test_saturation_level_tso_calculation_multi():
    # TBD: mock pando.tso_calculation()
    wl = np.logspace(0, 2, 1000)
    depth = [wl, np.tile(0.03, len(wl))]
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    aperture = ['ch1', 'ch2', 'ch3', 'ch4']
    tso = pando.tso_calculation(
        'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
        ngroup=250, aperture=aperture,
    )

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

