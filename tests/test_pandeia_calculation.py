# Copyright (c) 2024 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import gen_tso.pandeia_io as jwst
import numpy as np


def test_calc_saturation_single():
    instrument = 'nircam'
    mode = 'ssgrism'
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

