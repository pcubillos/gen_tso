# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import pickle
import numpy as np
import gen_tso.pandeia_io as jwst


def mock_perform_calculation_single():
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    result = pando.perform_calculation(
        ngroup=2, nint=683, readout='rapid', filter='f444w',
    )
    result['1d'] = {}
    result['2d'] = {}
    result['3d'] = {}
    with open('perform_calculation_nircam_lw_tsgrism.pkl', 'wb') as f:
        pickle.dump(result, f)


def mock_perform_calculation_multi():
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    aperture = ['ch1', 'ch2', 'ch3', 'ch4']
    results = pando.perform_calculation(
        ngroup=250, nint=40, aperture=aperture,
    )
    for result in results:
        result['1d'] = {}
        result['2d'] = {}
        result['3d'] = {}
    with open('perform_calculation_miri_mrs_ts.pkl', 'wb') as f:
        pickle.dump(results, f)


def mock_tso_calculation_multi():
    wl = np.logspace(0, 2, 1000)
    depth = [wl, np.tile(0.03, len(wl))]
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    tso = pando.tso_calculation(
        'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
        ngroup=90, readout='rapid', filter='f444w',
    )

    thin = 20
    tso['wl'] = tso['wl'][::thin]
    tso['depth_spectrum'] = tso['depth_spectrum'][::thin]
    tso['flux_in'] = tso['flux_in'][::thin]
    tso['flux_out'] = tso['flux_out'][::thin]
    tso['var_in'] = tso['var_in'][::thin]
    tso['var_out'] = tso['var_out'][::thin]

    for result in (tso['report_in'], tso['report_out']):
        result['1d'] = {}
        result['2d'] = {}
        result['3d'] = {}
        spec = result['input']['scene'][0]['spectrum']['sed']['spectrum']
        wl, spectrum = spec
        spec = wl[::2000], spectrum[::2000]
        result['input']['scene'][0]['spectrum']['sed']['spectrum'] = spec

    with open('tso_calculation_nircam_lw_tsgrism.pkl', 'wb') as f:
        pickle.dump(tso, f)


def mock_tso_calculation_multi():
    wl = np.logspace(0, 2, 1000)
    depth = [wl, np.tile(0.03, len(wl))]
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    aperture = ['ch1', 'ch2', 'ch3', 'ch4']
    tso = pando.tso_calculation(
        'transit', transit_dur=2.1, obs_dur=6.0, depth_model=depth,
        ngroup=250, aperture=aperture,
    )
    for report in tso:
        thin = 20
        report['wl'] = report['wl'][::thin]
        report['depth_spectrum'] = report['depth_spectrum'][::thin]
        report['flux_in'] = report['flux_in'][::thin]
        report['flux_out'] = report['flux_out'][::thin]
        report['var_in'] = report['var_in'][::thin]
        report['var_out'] = report['var_out'][::thin]

        for result in (report['report_in'], report['report_out']):
            result['1d'] = {}
            result['2d'] = {}
            result['3d'] = {}
            spec = result['input']['scene'][0]['spectrum']['sed']['spectrum']
            wl, spectrum = spec
            spec = wl[::2000], spectrum[::2000]
            result['input']['scene'][0]['spectrum']['sed']['spectrum'] = spec

    with open('tso_calculation_miri_mrs_ts.pkl', 'wb') as f:
        pickle.dump(tso, f)

