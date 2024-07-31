from gen_tso.pandeia_io import get_configs
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

