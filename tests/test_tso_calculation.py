# Copyright (c) 2025-2026 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import gen_tso.pandeia_io as jwst
import pyratbay.spectrum as ps


# These are the most important tests where I check that Gen TSO
# reproduces the ETC calculations.
# Values may differ from ETC because background is not exactly the same

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Spectroscopy
def test_tso_calculation_miri_lrsslitless():
    pando = jwst.PandeiaCalculation('miri', 'lrsslitless')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    ngroup = 34

    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=100)
    nwave = len(wl)
    depth = np.tile(0.0305, nwave)
    depth_model = [wl, depth]

    obs_type = 'transit'
    transit_dur = 2.1
    obs_dur = 6.0
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
    )

    obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        tso, resolution=19.0, n_obs=1, noiseless=True,
    )

    expected_wl = np.array([
         5.15701397,  5.43577148,  5.72959697,  6.03930491,  6.36575383,
         6.70984863,  7.07254315,  7.45484278,  7.85780725,  8.28255359,
         8.73025919,  9.20216509,  9.69957942, 10.22388101, 10.77652323,
        11.359038  , 11.97304005, 12.6202314 , 13.30240607, 13.75786506,
    ])
    expected_depth = np.array([
       0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305,
       0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305,
       0.0305, 0.0305, 0.0305, 0.0305,
    ])
    expected_error = np.array([
        3.27994822e-05, 3.33874215e-05, 3.00765731e-05, 3.27432531e-05,
        3.24328219e-05, 3.14196076e-05, 3.44537976e-05, 3.32943111e-05,
        3.79513508e-05, 3.82597560e-05, 4.14044811e-05, 4.50366019e-05,
        5.02078834e-05, 6.21737303e-05, 9.07213898e-05, 1.36271159e-04,
        2.31590099e-04, 4.49028105e-04, 8.88215410e-04, 2.85127248e-03,
    ])
    expected_hw = np.array([
        0.13571089, 0.14304662, 0.15077887, 0.15892908, 0.16751984,
        0.17657496, 0.18611956, 0.19618007, 0.2067844 , 0.21796194,
        0.22974366, 0.24216224, 0.25525209, 0.2690495 , 0.28359272,
        0.29892205, 0.31508   , 0.33211135, 0.35006332, 0.10539567,
    ])

    np.testing.assert_allclose(obs_wl, expected_wl)
    np.testing.assert_allclose(obs_depth, expected_depth)
    np.testing.assert_allclose(obs_error, expected_error)
    np.testing.assert_allclose(band_widths, expected_hw)


def test_tso_calculation_miri_mrs():
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    aperture = ['ch1', 'ch2', 'ch3', 'ch4']
    ngroup = 100

    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=100)
    nwave = len(wl)
    depth = np.tile(0.0305, nwave)
    depth_model = [wl, depth]

    obs_type = 'transit'
    transit_dur = 2.1
    obs_dur = 6.0
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
        aperture=aperture,
    )
    obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        tso[0], resolution=50.0, n_obs=1, noiseless=True,
    )
    # ', '.join([f'{e:.8e}' for e in obs_error])

    expected_wl = np.array([
        4.97040821, 5.0708205 , 5.17326132, 5.27777165, 5.3843933 ,
        5.49316892, 5.60414203, 5.70467786
    ])
    expected_depth = np.array([
        0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305
    ])
    expected_error = np.array([
        1.03376672e-04, 1.04049712e-04, 1.04388385e-04, 1.04640633e-04,
        1.07915262e-04, 1.05269953e-04, 1.04706705e-04, 1.19906211e-04,
    ])
    expected_hw = np.array([
        0.04970408, 0.0507082 , 0.05173261, 0.05277772, 0.05384393,
        0.05493169, 0.05604142, 0.04449441,
    ])

    np.testing.assert_allclose(obs_wl, expected_wl)
    np.testing.assert_allclose(obs_depth, expected_depth)
    np.testing.assert_allclose(obs_error, expected_error)
    np.testing.assert_allclose(band_widths, expected_hw)


def test_tso_calculation_nirspec_bots():
    pando = jwst.PandeiaCalculation('nirspec', 'bots')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    ngroup = 14

    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=100)
    nwave = len(wl)
    depth = np.tile(0.0305, nwave)
    depth_model = [wl, depth]

    obs_type = 'transit'
    transit_dur = 2.1
    obs_dur = 6.0
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
    )
    obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        tso, resolution=19.0, n_obs=1, noiseless=True,
    )

    expected_wl = np.array([
        2.94812634, 3.10748452, 3.27545665, 3.45250837, 3.63913044,
        3.83584019, 4.04318291, 4.26173333, 4.4920973 , 4.73491337,
        4.99085463, 5.14943535,
    ])
    expected_depth = np.array([
        0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305,
        0.0305, 0.0305, 0.0305, 0.0305,
    ])
    expected_error = np.array([
        1.27417578e-05, 1.16182446e-05, 1.14774916e-05, 1.15256531e-05,
        1.23562613e-05, 1.72944447e-05, 1.38014796e-05, 1.54449036e-05,
        1.85347764e-05, 2.07968605e-05, 2.31659012e-05, 5.51837375e-05,
    ])
    expected_hw = np.array([
        0.07758227, 0.08177591, 0.08619623, 0.09085548, 0.09576659,
        0.10094316, 0.10639955, 0.11215088, 0.11821309, 0.12460298,
        0.13133828, 0.02724244,
    ])

    np.testing.assert_allclose(obs_wl, expected_wl)
    np.testing.assert_allclose(obs_depth, expected_depth)
    np.testing.assert_allclose(obs_error, expected_error)
    np.testing.assert_allclose(band_widths, expected_hw)


def test_tso_calculation_niriss_soss_96():
    pando = jwst.PandeiaCalculation('niriss', 'soss')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    subarray = 'substrip96'
    ngroup = 7

    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=100)
    nwave = len(wl)
    depth = np.tile(0.0305, nwave)
    depth_model = [wl, depth]

    obs_type = 'transit'
    transit_dur = 2.1
    obs_dur = 6.0
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
        subarray=subarray,
    )
    obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        tso, resolution=19.0, n_obs=1, noiseless=True,
    )

    expected_wl = np.array([
        0.85335677, 0.89948416, 0.94810493, 0.99935384, 1.05337297,
        1.11031205, 1.17032892, 1.23358994, 1.30027048, 1.37055537,
        1.44463944, 1.52272806, 1.60503769, 1.69179648, 1.78324494,
        1.87963656, 1.98123853, 2.08833251, 2.20121535, 2.32019996,
        2.44561617, 2.57781164, 2.71715281, 2.79867846,
    ])
    expected_depth = np.array([
        0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305,
        0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305,
        0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305
    ])
    expected_error = np.array([
        1.80642096e-05, 1.47988572e-05, 1.29737230e-05, 1.15557558e-05,
        1.05266862e-05, 9.70211247e-06, 9.27297369e-06, 8.90963847e-06,
        8.90408055e-06, 9.08981799e-06, 9.43052290e-06, 9.72398475e-06,
        1.01331626e-05, 1.07969693e-05, 1.18418281e-05, 1.30949257e-05,
        1.44735349e-05, 1.56513620e-05, 1.70492345e-05, 1.92741418e-05,
        2.20525620e-05, 2.44189009e-05, 2.64790774e-05, 8.03396224e-05
    ])
    expected_hw = np.array([
        2.24567571e-02, 2.36706359e-02, 2.49501297e-02, 2.62987854e-02,
        2.77203413e-02, 2.92187382e-02, 3.07981294e-02, 3.24628932e-02,
        3.42176442e-02, 3.60672465e-02, 3.80168274e-02, 4.00717911e-02,
        4.22378338e-02, 4.45209600e-02, 4.69274984e-02, 4.94641199e-02,
        5.21378561e-02, 5.49561186e-02, 5.79267196e-02, 6.10578937e-02,
        6.43583203e-02, 6.78371485e-02, 7.15040214e-02, 1.00216259e-02
    ])

    np.testing.assert_allclose(obs_wl, expected_wl)
    np.testing.assert_allclose(obs_depth, expected_depth)
    np.testing.assert_allclose(obs_error, expected_error)
    np.testing.assert_allclose(band_widths, expected_hw)


def test_tso_calculation_niriss_soss_stripe204():
    pando = jwst.PandeiaCalculation('niriss', 'soss')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    subarray = 'sub204stripe_soss'
    ngroup = 25

    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=100)
    nwave = len(wl)
    depth = np.tile(0.0305, nwave)
    depth_model = [wl, depth]

    obs_type = 'transit'
    transit_dur = 2.1
    obs_dur = 6.0
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
        subarray=subarray,
    )
    obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        tso, resolution=19.0, n_obs=1, noiseless=True,
    )

    expected_wl = np.array([
        0.85335677, 0.89948416, 0.94810493, 0.99935384, 1.05337297,
        1.11031205, 1.17032892, 1.23358994, 1.30027048, 1.37055537,
        1.44463944, 1.52272806, 1.60503769, 1.69179648, 1.78324494,
        1.87963656, 1.98123853, 2.08833251, 2.20121535, 2.32019996,
        2.44561617, 2.57781164, 2.71715281, 2.79867846,
    ])
    expected_depth = np.array([
        0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305,
        0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305,
        0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305,
    ])
    expected_error = np.array([
        5.13600721e-05, 4.20797080e-05, 3.68916122e-05, 3.28604481e-05,
        2.99346945e-05, 2.75901881e-05, 2.63700131e-05, 2.53368314e-05,
        2.53208604e-05, 2.58487866e-05, 2.68171950e-05, 2.76511852e-05,
        2.88140953e-05, 3.07005713e-05, 3.36697124e-05, 3.72299757e-05,
        4.11459659e-05, 4.44905216e-05, 4.84588093e-05, 5.47736552e-05,
        6.26564010e-05, 6.93660156e-05, 7.52008492e-05, 2.27836546e-04,
    ])
    expected_hw = np.array([
        2.24567571e-02, 2.36706359e-02, 2.49501297e-02, 2.62987854e-02,
        2.77203413e-02, 2.92187382e-02, 3.07981294e-02, 3.24628932e-02,
        3.42176442e-02, 3.60672465e-02, 3.80168274e-02, 4.00717911e-02,
        4.22378338e-02, 4.45209600e-02, 4.69274984e-02, 4.94641199e-02,
        5.21378561e-02, 5.49561186e-02, 5.79267196e-02, 6.10578937e-02,
        6.43583203e-02, 6.78371485e-02, 7.15040214e-02, 1.00216259e-02
    ])

    np.testing.assert_allclose(obs_wl, expected_wl)
    np.testing.assert_allclose(obs_depth, expected_depth)
    np.testing.assert_allclose(obs_error, expected_error)
    np.testing.assert_allclose(band_widths, expected_hw)


def test_tso_calculation_nircam_lw_tsgrism():
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    ngroup = 90

    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=100)
    nwave = len(wl)
    depth = np.tile(0.0305, nwave)
    depth_model = [wl, depth]

    obs_type = 'transit'
    transit_dur = 2.1
    obs_dur = 6.0
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
    )
    obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        tso, resolution=25.0, n_obs=1, noiseless=True,
    )

    expected_wl = np.array([
        3.80213776, 3.95732705, 4.1188506 , 4.28696696, 4.4619452 ,
        4.64406541, 4.8336191 , 4.96435824,
    ])
    expected_depth = np.array([
        0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305,
    ])
    expected_error = np.array([
        8.00670255e-05, 1.96969569e-05, 1.96594281e-05, 2.10927730e-05,
        2.39498746e-05, 2.66036676e-05, 2.98726399e-05, 6.32197854e-05,
    ])
    expected_hw = np.array([
        0.07604276, 0.07914654, 0.08237701, 0.08573934, 0.0892389 ,
        0.09288131, 0.09667238, 0.03406676,
    ])

    np.testing.assert_allclose(obs_wl, expected_wl)
    np.testing.assert_allclose(obs_depth, expected_depth)
    np.testing.assert_allclose(obs_error, expected_error)
    np.testing.assert_allclose(band_widths, expected_hw)


def test_tso_calculation_nircam_sw_tsgrism():
    pando = jwst.PandeiaCalculation('nircam', 'sw_tsgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    aperture = 'dhs0spec8'
    filter = 'f150w'
    readout = 'dhs3'
    subarray = 'sub260s4_8-spectra'
    ngroup = 30

    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=100)
    nwave = len(wl)
    depth = np.tile(0.0305, nwave)
    depth_model = [wl, depth]

    obs_type = 'transit'
    transit_dur = 2.1
    obs_dur = 6.0
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
        aperture=aperture, filter=filter, readout=readout, subarray=subarray,
    )
    obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        tso, resolution=25.0, n_obs=1, noiseless=True,
    )

    expected_wl = np.array([
        1.29988773, 1.35294438, 1.4081666 , 1.46564278, 1.52546494,
        1.58772881, 1.65253407, 1.70608238
    ])
    expected_depth = np.array([
        0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305, 0.0305,
    ])
    expected_error = np.array([
        5.58927985e-04, 3.24816594e-05, 2.95266142e-05, 2.88022978e-05,
        2.78360135e-05, 2.78001363e-05, 3.44282112e-05, 3.23269323e-03,
    ])
    expected_hw = np.array([
        2.59977547e-02, 2.70588875e-02, 2.81633319e-02, 2.93128557e-02,
        3.05092987e-02, 3.17545762e-02, 3.30506814e-02, 2.04976252e-02,
    ])

    np.testing.assert_allclose(obs_wl, expected_wl)
    np.testing.assert_allclose(obs_depth, expected_depth)
    np.testing.assert_allclose(obs_error, expected_error)
    np.testing.assert_allclose(band_widths, expected_hw)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Photometry
def test_tso_calculation_nircam_sw_ts():
    pando = jwst.PandeiaCalculation('nircam', 'sw_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 11.0)

    filter = 'f140m'
    subarray = 'subgrism256'
    aperture = 'wlp8__tsgrism'
    ngroup = 100

    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=100)
    nwave = len(wl)
    depth = np.tile(0.0305, nwave)
    depth_model = [wl, depth]

    obs_type = 'transit'
    transit_dur = 2.1
    obs_dur = 6.0
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
        aperture=aperture, filter=filter, subarray=subarray,
    )
    obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        tso, resolution=25.0, n_obs=1, noiseless=True,
    )

    expected_wl = 1.40906336
    expected_depth = 0.03049999464010178
    expected_error = 0.0011131798407776421
    expected_hw = np.array([0.07906336022370586, 0.07093663977627762])

    np.testing.assert_allclose(obs_wl[0], expected_wl)
    np.testing.assert_allclose(obs_depth[0], expected_depth)
    np.testing.assert_allclose(obs_error[0], expected_error)
    np.testing.assert_allclose(band_widths[0], expected_hw)



def test_tso_calculation_nircam_lw_ts():
    pando = jwst.PandeiaCalculation('nircam', 'lw_ts')
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 10.055)
    filter = 'f480m'
    subarray = 'sub64p'
    ngroup = 50

    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=100)
    nwave = len(wl)
    depth = np.tile(0.0305, nwave)
    depth_model = [wl, depth]

    obs_type = 'transit'
    transit_dur = 2.1
    obs_dur = 6.0
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
        filter=filter, subarray=subarray,
    )
    obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        tso, n_obs=1, noiseless=True,
    )

    expected_wl = 4.80968392
    expected_depth = 0.03049999464010178
    expected_error = 4.19926025e-05
    expected_hw = np.array([0.15968392, 0.19531608])

    np.testing.assert_allclose(obs_wl[0], expected_wl)
    np.testing.assert_allclose(obs_depth[0], expected_depth)
    np.testing.assert_allclose(obs_error[0], expected_error)
    np.testing.assert_allclose(band_widths[0], expected_hw)


def test_tso_calculation_miri_imaging_ts():
    pando = jwst.PandeiaCalculation('miri', 'imaging_ts')
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 10.055)
    filter = 'f560w'
    subarray = 'sub256'
    ngroup = 10

    wl = ps.constant_resolution_spectrum(0.1, 50.0, resolution=100)
    nwave = len(wl)
    depth = np.tile(0.0305, nwave)
    depth_model = [wl, depth]

    obs_type = 'transit'
    transit_dur = 2.1
    obs_dur = 6.0
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model, ngroup,
        filter=filter, subarray=subarray,
    )
    obs_wl, obs_depth, obs_error, band_widths = jwst.simulate_tso(
        tso, n_obs=1, noiseless=True,
    )

    expected_wl = 5.59842396
    expected_depth = 0.03049999676984494
    expected_error = 4.02787839e-05
    expected_hw = np.array([0.56842396, 0.60657604])

    np.testing.assert_allclose(obs_wl[0], expected_wl)
    np.testing.assert_allclose(obs_depth[0], expected_depth)
    np.testing.assert_allclose(obs_error[0], expected_error)
    np.testing.assert_allclose(band_widths[0], expected_hw)

