# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import gen_tso.pandeia_io as jwst
import numpy as np


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
    # assert


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


def test_tso_calculation_single():
    pando = jwst.PandeiaCalculation('nirspec', 'bots')
    # Set the stellar scene:
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.637)
    depth_model = np.loadtxt('data/models/WASP80b_transit.dat', unpack=True)

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


def test_tso_calculation_multiple():
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.637)
    depth_model = np.loadtxt('data/models/WASP80b_transit.dat', unpack=True)

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
    np.testing.assert_almost_equal(pixel_rate, 1335.1112060546875)
    np.testing.assert_almost_equal(full_well, 58100.00314689601)


def test_calc_saturation_multiple():
    inst = 'miri'
    mode = 'mrs_ts'
    pando = jwst.PandeiaCalculation(inst, mode)    
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    ngroup = 2
    disperser = 'short'
    filter = None
    subarray = 'full'
    readout = 'fastr1'
    aperture = 'ch1 ch2 ch3 ch4'.split()

    pixel_rate, full_well = pando.get_saturation_values(
        disperser, filter, subarray, readout, ngroup, aperture,
    )
    expected_rate = [204.3777924, 109.1716537, 32.7559395, 4.5567689]
    expected_well = [
        193655.0135037985,
        193655.00840719877,
        193655.00325247002,
        193655.00765403986,
    ]
    np.testing.assert_almost_equal(pixel_rate, expected_rate)
    np.testing.assert_almost_equal(full_well, expected_well)

