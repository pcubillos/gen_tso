# Copyright (c) 2025-2026 Patricio Cubillos
# Gen TSO is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import gen_tso.pandeia_io as jwst


# These are the most important tests where I check that Gen TSO
# reproduces the ETC calculations.
# Values may differ from ETC because background is not exactly the same

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Spectroscopy
def test_perform_calculation_miri_lrsslitless():
    pando = jwst.PandeiaCalculation('miri', 'lrsslitless')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    ngroup = 34
    nint = 1379
    report = pando.perform_calculation(
        ngroup, nint,
    )

    rep = report['scalar']
    np.testing.assert_allclose(rep['sn'], 2195.608856258305)
    np.testing.assert_allclose(rep['extracted_flux'], 2789.5626155289738)
    np.testing.assert_allclose(rep['extracted_noise'], 1.2705189303539557)
    np.testing.assert_allclose(rep['brightest_pixel'], 28375.764)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.7779784393379979)


def test_perform_calculation_miri_mrs():
    pando = jwst.PandeiaCalculation('miri', 'mrs_ts')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    ngroup = 50
    nint = 450

    report = pando.perform_calculation(
        ngroup, nint,
    )

    rep = report['scalar']
    np.testing.assert_allclose(rep['sn'], 2119.360333)
    np.testing.assert_allclose(rep['extracted_flux'], 257.471385)
    np.testing.assert_allclose(rep['extracted_noise'], 0.121485, rtol=1e-5)
    np.testing.assert_allclose(rep['brightest_pixel'], 171.94064)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.123194, rtol=1e-5)


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
    np.testing.assert_allclose(rep['sn'], 5900.225218)
    np.testing.assert_allclose(rep['extracted_flux'], 6164.919829)
    np.testing.assert_allclose(rep['extracted_noise'], 1.044862, rtol=1e-6)
    np.testing.assert_allclose(rep['brightest_pixel'], 3957.697)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.7688891998046, rtol=1e-6)


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
    np.testing.assert_allclose(rep['sn'], 11241.443443)
    np.testing.assert_allclose(rep['extracted_flux'], 28751.78125)
    np.testing.assert_allclose(rep['extracted_noise'], 2.5576592005322834)
    np.testing.assert_allclose(rep['brightest_pixel'], 3474.604)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.8547525849609375)


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
    np.testing.assert_allclose(rep['sn'], 21065.94312735414)
    np.testing.assert_allclose(rep['extracted_flux'], 28751.783203125)
    np.testing.assert_allclose(rep['extracted_noise'], 1.3648467115526761)
    np.testing.assert_allclose(rep['brightest_pixel'], 3474.603)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.8691140372395832)


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
    np.testing.assert_allclose(rep['sn'], 5655.338207471038)
    np.testing.assert_allclose(rep['extracted_flux'], 5184.434512776988)
    np.testing.assert_allclose(rep['extracted_noise'], 0.9167328853874809)
    np.testing.assert_allclose(rep['brightest_pixel'], 2365.2615)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.6933147253586052)


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
    np.testing.assert_allclose(rep['sn'], 3389.2153192064393)
    np.testing.assert_allclose(rep['extracted_flux'], 2315.540456689559)
    np.testing.assert_allclose(rep['extracted_noise'], 0.6832084239580642)
    np.testing.assert_allclose(rep['brightest_pixel'], 472.61523)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.7687627493685801)



# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Photometry
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
    np.testing.assert_allclose(rep['sn'], 11484.19406522153)
    np.testing.assert_allclose(rep['extracted_flux'], 109402.84156062189)
    np.testing.assert_allclose(rep['extracted_noise'], 9.526383909858763)
    np.testing.assert_allclose(rep['brightest_pixel'], 15428.072)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.639108779794989)


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
    np.testing.assert_allclose(rep['sn'], 12158.286838910573)
    np.testing.assert_allclose(rep['extracted_flux'], 118558.08763090083)
    np.testing.assert_allclose(rep['extracted_noise'], 9.751216532536098)
    np.testing.assert_allclose(rep['brightest_pixel'], 14810.168)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.7813039110316264)


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
    np.testing.assert_allclose(rep['sn'], 9721.546933694628)
    np.testing.assert_allclose(rep['extracted_flux'], 178542.65733673942)
    np.testing.assert_allclose(rep['extracted_noise'], 18.36566325858236)
    np.testing.assert_allclose(rep['brightest_pixel'], 39392.395)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.6701965356432831)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Acquisition
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
    np.testing.assert_allclose(rep['sn'], 512.235691848597)
    np.testing.assert_allclose(rep['extracted_flux'], 451145.6269531249)
    np.testing.assert_allclose(rep['extracted_noise'], 880.7383673812236)
    np.testing.assert_allclose(rep['brightest_pixel'], 47211.22)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.38772416049159586)


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
    np.testing.assert_allclose(rep['sn'], 317.4217727873375)
    np.testing.assert_allclose(rep['extracted_flux'], 2659249.79173341)
    np.testing.assert_allclose(rep['extracted_noise'], 8377.65402285439)
    np.testing.assert_allclose(rep['brightest_pixel'], 507660.62)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.4815854755865567)


def test_perform_calculation_niriss_target_acq():
    pando = jwst.PandeiaCalculation('niriss', 'target_acq')
    pando.set_scene('phoenix', 'k5v', '2mass,ks', 8.351)

    readout = 'nis'
    ngroup = 3
    nint = 1
    report = pando.perform_calculation(ngroup, nint, readout=readout)

    rep = report['scalar']
    np.testing.assert_allclose(rep['sn'], 406.28849555689436)
    np.testing.assert_allclose(rep['extracted_flux'], 469535.39416503895)
    np.testing.assert_allclose(rep['extracted_noise'], 1155.6699224807064)
    np.testing.assert_allclose(rep['brightest_pixel'], 69075.45)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.5238221861979167)


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
    np.testing.assert_allclose(rep['sn'], 75.69717749018785)
    np.testing.assert_allclose(rep['extracted_flux'], 513007.89941406244)
    np.testing.assert_allclose(rep['extracted_noise'], 6777.10736943343)
    np.testing.assert_allclose(rep['brightest_pixel'], 275807.12)
    np.testing.assert_allclose(rep['fraction_saturation'], 0.19043421184615383)


