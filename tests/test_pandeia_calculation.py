# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import pytest
import re
import numpy as np
import gen_tso.pandeia_io as jwst
from gen_tso.utils import ROOT


@pytest.mark.skip(reason='TBD')
def test_perform_calculation():
    pando = jwst.PandeiaCalculation('niriss', 'soss')
    # Set the stellar scene:
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.637)

    # Set a NIRSpec observation
    disperser = 'gr700xd'
    filter = 'clear'
    readout = 'nisrapid'
    subarray = 'substrip96'
    aperture = 'soss'
    ngroup = 3
    nint = 984

    report = pando.perform_calculation(
        ngroup, nint, disperser, filter, subarray, readout, aperture,
    )
    # TBD: asserts


#def test_read_noise_variance():
#    report_config = {
#        'mode': '',
#        'aperture': '',
#    }
#    inputs = (
#        ('miri', 'mrs_ts', 'ch1'),
#        ('miri', 'lrsslitless', 'imager'),
#        ('nircam', 'lw_tsgrism', 'lw'),
#        ('nirspec', 'bots',  's1600a1'),
#        ('niriss', 'soss', 'soss'),
#    )
#    for input in inputs:
#        inst, mode, aperture = input
#        ins_config = get_instrument_config('jwst', inst)
#        pando = jwst.PandeiaCalculation(inst, mode)
#        #print(pando.calc['configuration']['instrument']['aperture'])
#        info_dict = ins_config['detector_config']
#        report_config['mode'] = mode
#        report_config['aperture'] = aperture
#
#        #report_config = report['input']['configuration']['instrument']
#        if report_config['mode'] == 'mrs_ts':
#            aperture = report_config['aperture']
#            aperture = ins_config['aperture_config'][aperture]['detector']
#
#        if aperture not in ins_config['detector_config']:
#            aperture = 'default'
#
#        read_noise = ins_config['detector_config'][aperture]['rn']
#        if isinstance(read_noise, dict):
#            read_noise = read_noise['default']
#        print(inst, mode, read_noise)


def test_get_scene_phoenix():
    pando = jwst.PandeiaCalculation('nirspec', 'bots')
    pando.set_scene(
        sed_type='phoenix', sed_model='k5v',
        norm_band='2mass,ks', norm_magnitude=8.351,
    )
    scene = pando.get_scene()
    expected_scene = {
        'sed_type': 'phoenix',
        'key': 'k5v',
        'normalization': 'photsys',
        'bandpass': '2mass,ks',
        'norm_flux': 8.351,
        'norm_fluxunit': 'vegamag',
    }
    for key, val in expected_scene.items():
        assert scene[key] == val


def test_get_scene_unset():
    pando = jwst.PandeiaCalculation('nirspec', 'bots')
    scene = pando.get_scene()
    expected_scene = {
        'sed_type': 'flat',
        'unit': 'fnu',
        'z': 0.0,
        'normalization': 'at_lambda',
        'norm_wave': 2.0,
        'norm_flux': 0.001,
        'norm_fluxunit': 'mjy',
        'norm_waveunit': 'microns',
    }
    for key, val in expected_scene.items():
        assert scene[key] == val


# Spectroscopy
def test_get_configs_miri_lrs(capsys):
    pando = jwst.PandeiaCalculation('miri', 'lrsslitless')
    pando.get_configs()

    captured = capsys.readouterr()
    expected_captured = """apertures: ['imager']
dispersers: ['p750l']
filters: ['']
subarrays: ['slitlessprism']
readout patterns: ['fastr1', 'slowr1']

"""
    assert captured.out == expected_captured


def test_get_configs_miri_mrs(capsys):
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.get_configs()

    captured = capsys.readouterr()
    expected_captured = """apertures: ['ch1', 'ch2', 'ch3', 'ch4']
dispersers: ['short', 'medium', 'long']
filters: ['']
subarrays: ['full']
readout patterns: ['fastr1', 'slowr1']

"""
    assert captured.out == expected_captured


def test_get_configs_nircam_lw_tsgrism(capsys):
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.get_configs()

    captured = capsys.readouterr()
    expected_captured = """apertures: ['lw']
dispersers: ['grismr']
filters: ['f277w', 'f322w2', 'f356w', 'f444w']
subarrays: ['full', 'subgrism128', 'subgrism256', 'subgrism64', 'full (noutputs=1)', 'subgrism128 (noutputs=1)', 'subgrism256 (noutputs=1)', 'subgrism64 (noutputs=1)', 'sub41s1_2-spectra', 'sub82s2_4-spectra', 'sub164s4_8-spectra', 'sub260s4_8-spectra']
readout patterns: ['rapid', 'bright1', 'bright2', 'shallow2', 'shallow4', 'medium2', 'medium8', 'mediumdeep2', 'mediumdeep8', 'deep2', 'deep8', 'dhs3', 'dhs4', 'dhs5', 'dhs6', 'dhs7']

"""
    assert captured.out == expected_captured


def test_get_configs_nircam_sw_tsgrism(capsys):
    pando = jwst.PandeiaCalculation('nircam', 'sw_tsgrism')
    pando.get_configs()

    captured = capsys.readouterr()
    expected_captured = """apertures: ['dhs0spec2', 'dhs0spec4', 'dhs0spec8', 'dhs0bright']
dispersers: ['dhs0']
filters: ['f070w', 'f090w', 'f115w', 'f150w', 'f150w2', 'f200w']
subarrays: ['sub41s1_2-spectra', 'sub82s2_4-spectra', 'sub164s4_8-spectra', 'sub260s4_8-spectra']
readout patterns: ['rapid', 'bright1', 'dhs3', 'dhs4', 'dhs5', 'dhs6', 'dhs7']

"""
    assert captured.out == expected_captured


def test_get_configs_nirspec_bots(capsys):
    pando = jwst.PandeiaCalculation('nirspec', 'bots')
    pando.get_configs()

    captured = capsys.readouterr()
    expected_captured = """apertures: ['s1600a1']
grating/filter pairs: ['g140h/f070lp', 'g140h/f100lp', 'g140m/f070lp', 'g140m/f100lp', 'g235h/f170lp', 'g235m/f170lp', 'g395h/f290lp', 'g395m/f290lp', 'prism/clear']
subarrays: ['sub512', 'sub512s', 'sub1024a', 'sub1024b', 'sub2048']
readout patterns: ['nrs', 'nrsrapid']

"""
    assert captured.out == expected_captured


def test_get_configs_niriss_soss(capsys):
    pando = jwst.PandeiaCalculation('niriss', 'soss')
    pando.get_configs()

    captured = capsys.readouterr()
    expected_captured = """apertures: ['soss']
dispersers: ['gr700xd']
filters: ['clear', 'f277w']
subarrays: ['substrip256', 'substrip96', 'sossfull', 'sub17stripe_soss', 'sub60stripe_soss', 'sub204stripe_soss', 'sub680stripe_soss']
readout patterns: ['nis', 'nisrapid']

"""
    assert captured.out == expected_captured

# Acquisition
def test_get_configs_miri_acq(capsys):
    pando = jwst.PandeiaCalculation('miri', 'target_acq')
    pando.get_configs()

    captured = capsys.readouterr()
    expected_captured = """apertures: ['imager']
dispersers: ['']
filters: ['f560w', 'f1000w', 'f1500w', 'fnd']
subarrays: ['full', 'brightsky', 'sub256', 'sub128', 'sub64', 'slitlessprism']
readout patterns: ['fast', 'fastgrpavg', 'fastgrpavg8', 'fastgrpavg16', 'fastgrpavg32', 'fastgrpavg64']

"""
    assert captured.out == expected_captured

def test_get_configs_nircam_acq(capsys):
    pando = jwst.PandeiaCalculation('nircam', 'target_acq')
    pando.get_configs()

    captured = capsys.readouterr()
    expected_captured = """apertures: ['lw']
dispersers: ['']
filters: ['f335m', 'f405n']
subarrays: ['sub32tats']
readout patterns: ['rapid', 'bright1', 'bright2', 'shallow2', 'shallow4', 'medium2', 'medium8', 'mediumdeep2', 'mediumdeep8', 'deep2', 'deep8']

"""
    assert captured.out == expected_captured


def test_get_configs_nirspec_acq(capsys):
    pando = jwst.PandeiaCalculation('nirspec', 'target_acq')
    pando.get_configs()

    captured = capsys.readouterr()
    expected_captured = """apertures: ['s1600a1']
dispersers: ['']
filters: ['f110w', 'f140x', 'clear']
subarrays: ['full', 'sub32', 'sub2048']
readout patterns: ['nrsrapid', 'nrsrapidd6']

"""
    assert captured.out == expected_captured


def test_get_configs_niriss_acq(capsys):
    pando = jwst.PandeiaCalculation('niriss', 'target_acq')
    pando.get_configs()

    captured = capsys.readouterr()
    expected_captured = """apertures: ['imager', 'nrm']
dispersers: ['']
filters: ['f480m']
subarrays: ['subtaami']
readout patterns: ['nis', 'nisrapid']

"""
    assert captured.out == expected_captured


def test_show_config(capsys):
    pando = jwst.PandeiaCalculation('nirspec', 'bots')
    pando.set_scene(
        sed_type='phoenix', sed_model='k5v',
        norm_band='2mass,ks', norm_magnitude=8.351,
    )
    pando.show_config()

    captured = capsys.readouterr()
    expected_captured = """Instrument configuration:
    instrument = 'nirspec'
    mode = 'bots'
    aperture = 's1600a1'
    disperser = 'g395h'
    filter = 'f290lp'
    readout pattern = 'nrsrapid'
    subarray = 'sub2048'

Scene configuration:
    sed_type = 'phoenix'
    key = 'k5v'
    normalization = 'photsys'
    bandpass = '2mass,ks'
    norm_flux = 8.351
    norm_fluxunit = 'vegamag'
"""
    assert captured.out == expected_captured


def test_show_config_soss(capsys):
    pando = jwst.PandeiaCalculation('niriss', 'soss')
    pando.set_scene(
        sed_type='phoenix', sed_model='k5v',
        norm_band='2mass,ks', norm_magnitude=8.351,
    )
    pando.show_config()

    captured = capsys.readouterr()
    expected_captured = """Instrument configuration:
    instrument = 'niriss'
    mode = 'soss'
    aperture = 'soss'
    disperser = 'gr700xd'
    filter = 'clear'
    readout pattern = 'nisrapid'
    subarray = 'substrip256'
    order = 1

Scene configuration:
    sed_type = 'phoenix'
    key = 'k5v'
    normalization = 'photsys'
    bandpass = '2mass,ks'
    norm_flux = 8.351
    norm_fluxunit = 'vegamag'
"""
    assert captured.out == expected_captured


def test_saturation_fraction_get_ngroup():
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene(
        sed_type='phoenix', sed_model='k5v',
        norm_band='2mass,ks', norm_magnitude=8.351,
    )
    ngroup = pando.saturation_fraction(fraction=80.0)
    assert ngroup == 104


def test_saturation_fraction_get_saturation():
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene(
        sed_type='phoenix', sed_model='k5v',
        norm_band='2mass,ks', norm_magnitude=8.351,
    )
    saturation = pando.saturation_fraction(ngroup=91)

    expected_saturation = 69.3642895274301
    expected_saturation = 69.36980131146873
    np.testing.assert_almost_equal(saturation, expected_saturation)


def test_saturation_fraction_with_flux_rate():
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene(
        sed_type='phoenix', sed_model='k5v',
        norm_band='2mass,ks', norm_magnitude=8.351,
    )
    flux_rate = 1400.2094016444962
    full_well = 58100.001867429375
    ngroup = pando.saturation_fraction(
        fraction=80.0, flux_rate=flux_rate, full_well=full_well,
    )
    assert ngroup == 97


def test_saturation_fraction_error_both_inputs():
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene(
        sed_type='phoenix', sed_model='k5v',
        norm_band='2mass,ks', norm_magnitude=8.351,
    )

    error = re.escape('Only one of fraction and ngroup must be defined')
    with pytest.raises(ValueError, match=error):
        saturation = pando.saturation_fraction(ngroup=91, fraction=80.0)


def test_saturation_fraction_error_no_inputs():
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene(
        sed_type='phoenix', sed_model='k5v',
        norm_band='2mass,ks', norm_magnitude=8.351,
    )

    error = re.escape('At least one of fraction and ngroup must be defined')
    with pytest.raises(ValueError, match=error):
        saturation = pando.saturation_fraction()


def test_saturation_fraction_no_guess_band(capsys):
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene(
        sed_type='phoenix', sed_model='k5v',
        norm_band='2mass,h', norm_magnitude=8.351,
    )

    ngroup = pando.saturation_fraction(fraction=80.0)
    assert ngroup is None
    #captured = capsys.readouterr()
    #expected_captured = 'Error, can only guess for Ks band ("2mass,ks")\n'
    #assert captured.out == expected_captured


def test_saturation_fraction_no_guess_sed_type(capsys):
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    ngroup = pando.saturation_fraction(fraction=80.0)
    assert ngroup is None
    #captured = capsys.readouterr()
    #expected_captured = 'Error, can only guess for phoenix or kurucz SEDs\n'
    #assert captured.out == expected_captured


def test_saturation_fraction_no_guess_label(capsys):
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene(
        sed_type='phoenix', sed_model='k5v',
        norm_band='2mass,ks', norm_magnitude=8.351,
    )
    pando.calc['configuration']['instrument']['filter'] = 'bad_filter'

    ngroup = pando.saturation_fraction(fraction=80.0)
    assert ngroup is None
    #captured = capsys.readouterr()
    #expected_captured = "Error, no flux_rate spline for configuration label: 'lw_tsgrism_bad_filter_phoenix_k5v'\n"
    #assert captured.out == expected_captured


@pytest.mark.skip(reason='TBD')
def test_tso_calculation_single():
    pando = jwst.PandeiaCalculation('nirspec', 'bots')
    # Set the stellar scene:
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.637)
    model_file = f'{ROOT}data/models/WASP80b_transit.dat'
    depth_model = np.loadtxt(model_file, unpack=True)

    # Set a NIRSpec observation
    disperser = 'g395h'
    filter = 'f290lp'
    readout = 'nrsrapid'
    subarray = 'sub2048'
    ngroup = 16

    transit_dur = 2.753
    obs_dur = 7.1
    obs_type = 'transit'
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model,
        ngroup, disperser, filter, subarray, readout,
    )
    # assert


@pytest.mark.skip(reason='TBD')
def test_tso_calculation_multiple():
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.637)
    model_file = f'{ROOT}data/models/WASP80b_transit.dat'
    depth_model = np.loadtxt(model_file, unpack=True)

    disperser = 'short'
    filter = None
    readout = 'fastr1'
    subarray = 'full'
    ngroup = 16
    aperture = ['ch1', 'ch2']

    transit_dur = 2.753
    obs_dur = 7.1
    obs_type = 'transit'
    tso = pando.tso_calculation(
        obs_type, transit_dur, obs_dur, depth_model,
        ngroup, disperser, filter, subarray, readout, aperture,
    )
    # assert


def test_calc_saturation_single():
    instrument = 'nircam'
    mode = 'lw_tsgrism'
    pando = jwst.PandeiaCalculation(instrument, mode)
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 8.351)
    pixel_rate, full_well = pando.get_saturation_values(
        disperser='grismr', filter='f444w',
        readout='rapid', subarray='subgrism64',
    )
    np.testing.assert_almost_equal(pixel_rate, 1243.1856689453125)
    np.testing.assert_almost_equal(full_well, 58100.00)


def test_calc_saturation_multiple():
    inst = 'miri'
    mode = 'mrs_ts'
    pando = jwst.PandeiaCalculation(inst, mode)    
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 8.351)
    ngroup = 2
    disperser = 'short'
    filter = None
    subarray = 'full'
    readout = 'fastr1'
    aperture = 'ch1 ch2 ch3 ch4'.split()

    pixel_rate, full_well = pando.get_saturation_values(
        disperser, filter, subarray, readout, ngroup, aperture,
    )
    expected_rate = [163.1600037,  89.8821259,  28.5365067,   4.2614117]
    expected_well = [193655.0, 193655.0, 193655.0, 193655.0]
    np.testing.assert_almost_equal(pixel_rate, expected_rate)
    np.testing.assert_almost_equal(full_well, expected_well)

