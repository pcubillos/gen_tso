from gen_tso.pandeia_io import get_configs
import gen_tso.pandeia_io as jwst

import numpy as np
import pytest


instruments = [
    'miri',
    'nircam',
    'niriss',
    'nirspec',
]


def test_get_configs_miri_lrsslitless():
    configs = get_configs(instrument='miri', obs_type='spectroscopy')
    assert len(configs) == 2
    inst = configs[0]
    assert inst['mode'] == 'lrsslitless'
    assert inst['mode_label'] == 'Low Resolution Spectroscopy (LRS) Slitless'


def test_get_configs_miri_mrs_ts():
    configs = get_configs(instrument='miri', obs_type='spectroscopy')
    assert len(configs) == 2
    inst = configs[1]
    assert inst['mode'] == 'mrs_ts'
    assert inst['mode_label'] == 'MRS Time Series'


def test_get_configs_nircam_lw_tsgrism():
    configs = get_configs(instrument='nircam', obs_type='spectroscopy')
    assert len(configs) == 2
    inst = configs[0]
    assert inst['mode'] == 'lw_tsgrism'
    assert inst['mode_label'] == 'LW Grism Time Series'


def test_get_configs_nircam_sw_tsgrism():
    configs = get_configs(instrument='nircam', obs_type='spectroscopy')
    assert len(configs) == 2
    inst = configs[1]
    assert inst['mode'] == 'sw_tsgrism'
    assert inst['mode_label'] == 'SW Grism Time Series'


def test_get_configs_niriss_soss():
    configs = get_configs(instrument='niriss', obs_type='spectroscopy')
    assert len(configs) == 1
    inst = configs[0]
    assert inst['mode'] == 'soss'
    assert inst['mode_label'] == 'SOSS'


def test_get_configs_nirspec_bots():
    configs = get_configs(instrument='nirspec', obs_type='spectroscopy')
    assert len(configs) == 1
    inst = configs[0]
    assert inst['mode'] == 'bots'
    assert inst['mode_label'] == 'Bright Object Time Series'


@pytest.mark.parametrize('instrument', instruments)
def test_get_configs_acquisition(instrument):
    configs = get_configs(instrument=instrument, obs_type='acquisition')
    assert len(configs) == 1
    inst = configs[0]
    assert inst['mode'] == 'target_acq'


def test_load_flux_rate_splines_all():
    # Extract all splines:
    flux_rate_splines, full_wells = jwst.load_flux_rate_splines()
    # Evaluate for one config:
    obs_label = 'lw_tsgrism_f444w_phoenix_k5v'
    spline = flux_rate_splines[obs_label]
    flux_rate = spline(8.351)
    full_well = full_wells[obs_label]

    expected_full_well = 58100.001867429375
    expected_flux_rate = 3.1140133020415353
    np.testing.assert_allclose(flux_rate, expected_flux_rate)
    np.testing.assert_allclose(full_well, expected_full_well)


def test_load_flux_rate_splines_single():
    obs_label = 'lw_tsgrism_f444w_phoenix_k5v'
    flux_rate_spline, full_well = jwst.load_flux_rate_splines(obs_label)
    flux_rate = flux_rate_spline(8.351)

    expected_full_well = 58100.001867429375
    expected_flux_rate = 3.1140133020415353
    np.testing.assert_allclose(flux_rate, expected_flux_rate)
    np.testing.assert_allclose(full_well, expected_full_well)


def test_load_flux_rate_splines_not_found():
    obs_label = 'nope_lw_tsgrism_f444w_phoenix_k5v'
    flux_rate_spline, full_well = jwst.load_flux_rate_splines(obs_label)
    assert flux_rate_spline is None
    assert full_well is None

