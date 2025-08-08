# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import gen_tso.pandeia_io as jwst

import numpy as np
import pytest
import re



def test_get_instruments():
    instruments = jwst.get_instruments()

    expected_instruments = [
        'miri',
        'nircam',
        'niriss',
        'nirspec',
    ]
    assert instruments == expected_instruments


def test_get_modes_miri():
    modes = jwst.get_modes('miri')

    expected_modes = [
        'lrsslitless',
        'lrsslit',
        'mrs_ts',
        'imaging_ts',
        'target_acq',
    ]
    assert modes == expected_modes


def test_get_modes_nircam():
    modes = jwst.get_modes('nircam')

    expected_modes = [
        'lw_tsgrism',
        'sw_tsgrism',
        'lw_ts',
        'sw_ts',
        'target_acq',
    ]
    assert modes == expected_modes


def test_get_modes_nirspec():
    modes = jwst.get_modes('nirspec')

    expected_modes = [
        'bots',
        'target_acq',
    ]
    assert modes == expected_modes


def test_get_modes_niriss():
    modes = jwst.get_modes('niriss')

    expected_modes = [
        'soss',
        'target_acq',
    ]
    assert modes == expected_modes


def test_get_modes_type_spectro():
    modes = jwst.get_modes('miri', type='spectroscopy')
    expected_modes = [
        'lrsslitless',
        'lrsslit',
        'mrs_ts',
    ]
    assert modes == expected_modes


def test_get_modes_type_photo():
    modes = jwst.get_modes('miri', type='photometry')
    expected_modes = [
        'imaging_ts',
    ]
    assert modes == expected_modes


def test_get_modes_type_acq():
    expected_modes = ['target_acq']
    for inst in jwst.get_instruments():
        modes = jwst.get_modes(inst, type='acquisition')
        assert modes == expected_modes


def test_get_sed_types():
    sed_types = jwst.get_sed_types()
    expected_sed_types = [
        'phoenix',
        'k93models',
        'bt_settl',
    ]
    assert sed_types == expected_sed_types


def test_get_throughputs_all():
    throughputs = jwst.get_throughputs()

    assert list(throughputs) == ['spectroscopy', 'photometry', 'acquisition']

    expected_inst = ['miri', 'nircam', 'niriss', 'nirspec']
    assert list(throughputs['spectroscopy']) == expected_inst

    assert list(throughputs['spectroscopy']['nirspec']) == ['bots']

    subarrays = throughputs['spectroscopy']['nirspec']['bots']
    expected_subs = ['sub512', 'sub512s', 'sub1024a', 'sub1024b', 'sub2048']
    assert list(subarrays) == expected_subs

    filters = list(throughputs['spectroscopy']['nirspec']['bots']['sub2048'])
    expected_filters = [
        'g140m/f070lp',
        'g140h/f070lp',
        'g140m/f100lp',
        'g140h/f100lp',
        'g235m/f170lp', 
        'g235h/f170lp',
        'g395m/f290lp',
        'g395h/f290lp',
        'prism/clear',
    ]
    assert list(filters) == expected_filters


def test_get_throughputs_type():
    throughputs = jwst.get_throughputs(type='acquisition')
    assert list(throughputs) == ['miri', 'nircam', 'niriss', 'nirspec']
    assert list(throughputs['nirspec']) == ['target_acq']


def test_get_throughputs_inst():
    throughputs = jwst.get_throughputs(inst='nirspec')
    assert list(throughputs) == ['bots', 'target_acq']

    expected_subs = ['sub512', 'sub512s', 'sub1024a', 'sub1024b', 'sub2048']
    assert list(throughputs['bots']) == expected_subs


def test_get_throughputs_mode():
    throughputs = jwst.get_throughputs(inst='nirspec', mode='bots')
    assert list(throughputs) == ['sub512', 'sub512s', 'sub1024a', 'sub1024b', 'sub2048']


def test_get_throughputs_mode_missing_inst():
    error = 'Also need to specify an instrument when requesting a specific mode'
    with pytest.raises(ValueError, match=re.escape(error)):
        throughputs = jwst.get_throughputs(mode='bots')



def test_generate_all_instruments():
    detectors = jwst.generate_all_instruments()
    assert len(detectors) == 14
    # TBD: what to test?



def test_get_configs_miri_lrsslitless():
    configs = jwst._get_configs(instrument='miri', obs_type='spectroscopy')
    assert len(configs) == 3
    inst = configs[0]
    assert inst['mode'] == 'lrsslitless'
    assert inst['mode_label'] == 'Low Resolution Spectroscopy (LRS) Slitless'


def test_get_configs_miri_mrs_ts():
    configs = jwst._get_configs(instrument='miri', obs_type='spectroscopy')
    inst = configs[2]
    assert inst['mode'] == 'mrs_ts'
    assert inst['mode_label'] == 'MRS Time Series'


def test_get_configs_nircam_lw_tsgrism():
    configs = jwst._get_configs(instrument='nircam', obs_type='spectroscopy')
    inst = configs[0]
    assert inst['mode'] == 'lw_tsgrism'
    assert inst['mode_label'] == 'LW Grism Time Series'


def test_get_configs_nircam_sw_tsgrism():
    configs = jwst._get_configs(instrument='nircam', obs_type='spectroscopy')
    assert len(configs) == 2
    inst = configs[1]
    assert inst['mode'] == 'sw_tsgrism'
    assert inst['mode_label'] == 'SW Grism Time Series'


def test_get_configs_niriss_soss():
    configs = jwst._get_configs(instrument='niriss', obs_type='spectroscopy')
    assert len(configs) == 1
    inst = configs[0]
    assert inst['mode'] == 'soss'
    assert inst['mode_label'] == 'SOSS'


def test_get_configs_nirspec_bots():
    configs = jwst._get_configs(instrument='nirspec', obs_type='spectroscopy')
    assert len(configs) == 1
    inst = configs[0]
    assert inst['mode'] == 'bots'
    assert inst['mode_label'] == 'Bright Object Time Series'


def test_get_configs_acquisition():
    for instrument in jwst.get_instruments():
        configs = jwst._get_configs(instrument=instrument, obs_type='acquisition')
        assert len(configs) == 1
        inst = configs[0]
        assert inst['mode'] == 'target_acq'


def test_load_flux_rate_splines_all():
    # Extract all splines:
    flux_rate_splines, full_wells = jwst._load_flux_rate_splines()
    # Evaluate for one config:
    obs_label = 'lw_tsgrism_f444w_phoenix_k5v'
    spline = flux_rate_splines[obs_label]
    flux_rate = spline(8.351)
    full_well = full_wells[obs_label]

    expected_full_well = 58100.001867429375
    expected_flux_rate = 3.114047796362145
    np.testing.assert_allclose(flux_rate, expected_flux_rate)
    np.testing.assert_allclose(full_well, expected_full_well)


def test_load_flux_rate_splines_single():
    obs_label = 'lw_tsgrism_f444w_phoenix_k5v'
    flux_rate_spline, full_well = jwst._load_flux_rate_splines(obs_label)
    flux_rate = flux_rate_spline(8.351)

    expected_full_well = 58100.001867429375
    expected_flux_rate = 3.114047796362145
    np.testing.assert_allclose(flux_rate, expected_flux_rate)
    np.testing.assert_allclose(full_well, expected_full_well)


def test_load_flux_rate_splines_not_found():
    obs_label = 'nope_lw_tsgrism_f444w_phoenix_k5v'
    flux_rate_spline, full_well = jwst._load_flux_rate_splines(obs_label)
    assert flux_rate_spline is None
    assert full_well is None

