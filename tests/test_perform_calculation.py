# Copyright (c) 2025 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import gen_tso.pandeia_io as jwst


# These are the most important tests where I check that Gen TSO
# reproduces the ETC calculations

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Spectroscopy  (workbook 261611)
def test_perform_calculation_miri_lrsslitless():
    pando = jwst.PandeiaCalculation('miri', 'lrsslitless')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    ngroup = 34
    nint = 1379
    report = pando.perform_calculation(
        ngroup, nint,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 2216.858286)
    np.testing.assert_allclose(rep['extracted_flux'], 2789.8673)
    np.testing.assert_allclose(rep['extracted_noise'], 1.258478, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 28412.377)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.793349, rtol=1e-6)


def test_perform_calculation_miri_lrsslit():
    pando = jwst.PandeiaCalculation('miri', 'lrsslit')
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 10.055)

    ngroup = 8
    nint = 713
    report = pando.perform_calculation(
        ngroup, nint,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 606.9103022719474)
    np.testing.assert_allclose(rep['extracted_flux'], 306.7919211072417)
    np.testing.assert_allclose(rep['extracted_noise'], 0.5054979623163043)
    np.testing.assert_allclose(rep['brightest_pixel'], 6393.106)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.7328961195941236)


def test_perform_calculation_miri_mrs():
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    ngroup = 50
    nint = 450

    report = pando.perform_calculation(
        ngroup, nint,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 2133.8171597)
    np.testing.assert_allclose(rep['extracted_flux'], 256.7128099)
    np.testing.assert_allclose(rep['extracted_noise'], 0.120307, rtol=1e-5)
    np.testing.assert_allclose(rep['brightest_pixel'], 172.13635)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.123334, rtol=1e-6)


def test_perform_calculation_nirspec_bots():
    pando = jwst.PandeiaCalculation('nirspec', 'bots')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    pando.calc['background'] = 'ecliptic'

    ngroup = 14
    nint = 550

    report = pando.perform_calculation(
        ngroup, nint,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 5909.949946609753)
    np.testing.assert_allclose(rep['extracted_flux'], 6185.192697986513)
    np.testing.assert_allclose(rep['extracted_noise'], 1.04657277199, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 3976.0142)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.7724477, rtol=1e-6)


def test_perform_calculation_niriss_soss_96():
    pando = jwst.PandeiaCalculation('niriss', 'soss')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    subarray = 'substrip96'
    ngroup = 8
    nint = 350

    report = pando.perform_calculation(
        ngroup, nint, subarray=subarray,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 11269.247048301882)
    np.testing.assert_allclose(rep['extracted_flux'], 28856.867285530574)
    np.testing.assert_allclose(rep['extracted_noise'], 2.560673944039491)
    np.testing.assert_allclose(rep['brightest_pixel'], 3191.4676620684377)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.7851010448688357)


def test_perform_calculation_niriss_soss_stripe204():
    pando = jwst.PandeiaCalculation('niriss', 'soss')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    subarray = 'sub204stripe_soss'
    ngroup = 32
    nint = 1165

    report = pando.perform_calculation(
        ngroup, nint, subarray=subarray,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 21117.856800942976)
    np.testing.assert_allclose(rep['extracted_flux'], 28856.8671980996)
    np.testing.assert_allclose(rep['extracted_noise'], 1.3664676046487376)
    np.testing.assert_allclose(rep['brightest_pixel'], 3191.466522334738)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.798292159453329)


def test_perform_calculation_nircam_lw_tsgrism():
    pando = jwst.PandeiaCalculation('nircam', 'lw_tsgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)
    pando.calc['background'] = 'ecliptic'

    filter = 'f322w2'
    readout = 'rapid'
    subarray = 'subgrism64'
    ngroup = 50
    nint = 442

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray, readout=readout,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 5663.234922906078)
    np.testing.assert_allclose(rep['extracted_flux'], 5197.941289562555)
    np.testing.assert_allclose(rep['extracted_noise'], 0.9178396023337207)
    np.testing.assert_allclose(rep['brightest_pixel'], 2370.5247)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.6948574903877508)


def test_perform_calculation_nircam_sw_tsgrism():
    pando = jwst.PandeiaCalculation('nircam', 'sw_tsgrism')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    aperture = 'dhs0spec8'
    filter = 'f150w'
    readout = 'dhs3'
    subarray = 'sub260s4_8-spectra'
    ngroup = 30
    nint = 62

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray, readout=readout,
        aperture=aperture,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 3399.893684600709)
    np.testing.assert_allclose(rep['extracted_flux'], 2326.9298324581123)
    np.testing.assert_allclose(rep['extracted_noise'], 0.6844125282498038)
    np.testing.assert_allclose(rep['brightest_pixel'], 485.7248)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.7900869454569583)



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Photometry (workbook 261613)
def test_perform_calculation_nircam_sw_ts():
    pando = jwst.PandeiaCalculation('nircam', 'sw_ts')
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 10.055)

    filter = 'f212n'
    readout = 'rapid'
    subarray = 'sub160p'
    ngroup = 11
    nint = 500

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray, readout=readout,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 10907.381441388885)
    np.testing.assert_allclose(rep['extracted_flux'], 95399.91842956156)
    np.testing.assert_allclose(rep['extracted_noise'], 8.746363088354034)
    np.testing.assert_allclose(rep['brightest_pixel'], 15469.762)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.6408357677853426)


def test_perform_calculation_nircam_lw_ts():
    pando = jwst.PandeiaCalculation('nircam', 'lw_ts')
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 10.055)

    filter = 'f480m'
    readout = 'rapid'
    subarray = 'sub160p'
    ngroup = 11
    nint = 500

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray, readout=readout,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 9885.3342620672)
    np.testing.assert_allclose(rep['extracted_flux'], 73134.03817783193)
    np.testing.assert_allclose(rep['extracted_noise'], 7.398236239564276)
    np.testing.assert_allclose(rep['brightest_pixel'], 14831.353)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.7824214937405873)


def test_perform_calculation_miri_imaging_ts():
    pando = jwst.PandeiaCalculation('miri', 'imaging_ts')
    pando.set_scene('phoenix', 'k2v', '2mass,ks', 10.055)

    filter = 'f560w'
    subarray = 'sub256'
    ngroup = 11
    nint = 500

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 9724.465601868)
    np.testing.assert_allclose(rep['extracted_flux'], 178300.55512741362)
    np.testing.assert_allclose(rep['extracted_noise'], 18.335254853815655)
    np.testing.assert_allclose(rep['brightest_pixel'], 39420.38)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.6706726435671684)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Acquisition (workbook 261612)
def test_perform_calculation_miri_target_acq():
    pando = jwst.PandeiaCalculation('miri', 'target_acq')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    filter = 'f1000w'
    subarray = 'slitlessprism'
    ngroup = 10
    nint = 1

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 512.3369780799846)
    np.testing.assert_allclose(rep['extracted_flux'], 451308.7021484374)
    np.testing.assert_allclose(rep['extracted_noise'], 880.8825469513158)
    np.testing.assert_allclose(rep['brightest_pixel'], 47220.94)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.3878040082233869)


def test_perform_calculation_nircam_target_acq():
    pando = jwst.PandeiaCalculation('nircam', 'target_acq')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    filter = 'f335m'
    ngroup = 5
    nint = 1

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter,
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 317.9818678861534)
    np.testing.assert_allclose(rep['extracted_flux'], 2667698.154677334)
    np.testing.assert_allclose(rep['extracted_noise'], 8389.466268662985)
    np.testing.assert_allclose(rep['brightest_pixel'], 510198.56)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.48399305611921367)


def test_perform_calculation_niriss_target_acq():
    pando = jwst.PandeiaCalculation('niriss', 'target_acq')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    readout = 'nis'
    ngroup = 3
    nint = 1
    report = pando.perform_calculation(ngroup, nint, readout=readout)

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 406.4132996199974)
    np.testing.assert_allclose(rep['extracted_flux'], 469783.7031249999)
    np.testing.assert_allclose(rep['extracted_noise'], 1155.9260082390383)
    np.testing.assert_allclose(rep['brightest_pixel'], 69039.6)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.5235503118489584)


def test_perform_calculation_nirspec_target_acq():
    pando = jwst.PandeiaCalculation('nirspec', 'target_acq')
    pando.set_scene('phoenix', 'k2v', 'gaia,g', 14.71)

    filter = 'f110w'
    subarray = 'sub32'
    readout = 'nrsrapid'
    ngroup = 3
    nint = 1

    report = pando.perform_calculation(
        ngroup, nint,
        filter=filter, subarray=subarray, readout=readout
    )

    rep = report['scalar']
    # Differ from ETC because background is not exactly the same
    np.testing.assert_allclose(rep['sn'], 76.96948861914326)
    np.testing.assert_allclose(rep['extracted_flux'], 526935.5126953124)
    np.testing.assert_allclose(rep['extracted_noise'], 6846.031098149417)
    np.testing.assert_allclose(rep['brightest_pixel'], 283687.75)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.19587548030769228)


